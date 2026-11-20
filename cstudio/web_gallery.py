"""Page through local gallery outputs and serve bounded PNG thumbnails."""
from __future__ import annotations

import io
import json
import math
import os
import re
import stat
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import HTTPException
from PIL import Image
from remote.job_paths import confined_path, relative_parts
from remote.manifest import MAX_REFERENCES, MAX_SEED, MODES, RESOLUTIONS, VARIANTS
from remote.models import qwen_image_21 as Q

from .colab_client import GPUS

JOB_NAME = re.compile(r"[A-Za-z0-9_.-]{1,100}")
FILE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,120}")
THUMBNAIL_EDGE = 192
THUMBNAIL_MAX_SIDE = 8192
THUMBNAIL_MAX_PIXELS = 4096**2
THUMBNAIL_CACHE_SIZE = 128
THUMBNAIL_LOCK = Lock()

# Nanosecond timestamp and artifact ID give a total order even for tied mtimes.
Position = tuple[int, str]


def safe_job_dir(jobs: Path, job_dir: str) -> Path:
    if not JOB_NAME.fullmatch(job_dir) or job_dir in (".", ".."):
        raise HTTPException(status_code=400, detail=f"不合法的任務資料夾名稱：{job_dir!r}")
    try:
        return confined_path(jobs, job_dir, directory=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="找不到任務資料夾") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="不允許存取此任務資料夾") from exc


def safe_subfile(jobs: Path, job_dir: str, subfolder: str, filename: str) -> Path:
    if subfolder not in ("out", "inputs"):
        raise HTTPException(status_code=400, detail="不支援的子資料夾")
    if not FILE_NAME.fullmatch(filename) or filename in (".", ".."):
        raise HTTPException(status_code=400, detail=f"不合法的檔案名稱：{filename!r}")
    if subfolder == "out" and not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="輸出作品必須是 PNG")
    base = safe_job_dir(jobs, job_dir)
    try:
        return confined_path(base, f"{subfolder}/{filename}")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"找不到圖片檔案：{job_dir}/{subfolder}/{filename}") from exc


def parse_cursor(cursor: str | None) -> Position | None:
    if cursor is None:
        return None
    timestamp, _, artifact_id = cursor.partition(":")
    try:
        job, subfolder, filename = relative_parts(artifact_id)
        if (not re.fullmatch(r"-?[0-9]{1,20}", timestamp) or not JOB_NAME.fullmatch(job)
                or subfolder != "out" or not FILE_NAME.fullmatch(filename) or not filename.lower().endswith(".png")):
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="不合法的畫廊分頁游標") from exc
    return int(timestamp), artifact_id


def inspect_image_meta(img_path: Path) -> dict[str, Any]:
    try:
        with Image.open(img_path, formats=("PNG",)) as im:
            has_alpha = im.mode in ("RGBA", "LA") or "transparency" in im.info
            return {"width": im.width, "height": im.height, "mode": im.mode, "has_alpha": has_alpha}
    except (OSError, ValueError, SyntaxError, OverflowError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return {"width": None, "height": None, "mode": None, "has_alpha": False}


def read_gallery_metadata(folder: Path, relative: str) -> dict[str, Any]:
    try:
        data = json.loads(confined_path(folder, relative).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def _candidates(jobs: Path, before: Position | None) -> list[Position]:
    """Read only directory entries and file metadata until the page is ordered."""
    if jobs.is_symlink() or jobs.is_junction():
        return []
    positions: list[Position] = []
    try:
        with os.scandir(jobs) as directories:
            for folder in directories:
                if not JOB_NAME.fullmatch(folder.name):
                    continue
                try:
                    out = confined_path(jobs, f"{folder.name}/out", directory=True)
                    with os.scandir(out) as images:
                        for entry in images:
                            name = entry.name
                            if (not FILE_NAME.fullmatch(name) or not name.lower().endswith(".png")
                                    or name.startswith(".") or name.lower().endswith("_sheet.png")):
                                continue
                            try:
                                info = entry.stat(follow_symlinks=False)
                            except OSError:
                                continue
                            if not stat.S_ISREG(info.st_mode) or getattr(info, "st_reparse_tag", 0) or not info.st_size:
                                continue
                            position = info.st_mtime_ns, f"{folder.name}/out/{name}"
                            if before is None or position < before:
                                positions.append(position)
                except (OSError, ValueError):
                    continue
    except OSError:
        pass
    return sorted(positions, reverse=True)


def _text(value) -> str | None:
    if isinstance(value, str) and value:
        # JSON can encode isolated surrogates that cannot be sent as UTF-8.
        return value.encode("utf-8", errors="replace").decode("utf-8")
    return None


def _choice(options, default, *values, normalize=str.lower):
    for value in values:
        if isinstance(value, str):
            value = normalize(value.strip())
            if value in options:
                return value
    return default


def _bounded_int(value, maximum: int) -> int | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value.strip()):
        value = int(value)
    return value if type(value) is int and 0 <= value <= maximum else None


def _number(value) -> int | float | None:
    try:
        if type(value) in (int, float) and math.isfinite(value):
            return value
    except OverflowError:
        pass
    return None


def _boolean(value, fallback: bool) -> bool:
    return value if type(value) is bool else fallback


def _input_artifacts(jobs: Path, job_dir: str, value) -> list[dict[str, str]]:
    inputs = []
    for relative in value[:MAX_REFERENCES] if isinstance(value, list) else []:
        try:
            parts = relative_parts(relative)
            if len(parts) != 2 or parts[0] != "inputs":
                continue
            name = safe_subfile(jobs, job_dir, *parts).name
        except (HTTPException, ValueError):
            continue
        artifact_id = f"{job_dir}/inputs/{name}"
        inputs.append({"filename": name, "artifact_id": artifact_id, "url": f"/api/jobs/{artifact_id}"})
    return inputs


def _item(jobs, png, position, info, manifest, results, m_jobs, r_jobs):
    mtime_ns, artifact_id = position
    job_dir, _, name = artifact_id.split("/")
    jid = png.stem
    if m_jobs and jid not in m_jobs and jid not in r_jobs:
        return None
    m_info = m_jobs.get(jid, {})
    record = r_jobs.get(jid)
    r_info = record if isinstance(record, dict) else {}
    if isinstance(r_info.get("status"), str) and r_info["status"] != "done":
        return None
    img = inspect_image_meta(png)
    if img["width"] is None:
        return None
    mode = _choice(MODES, "t2i", m_info.get("mode"), r_info.get("mode"))
    quant = _choice(Q.VALID_QUANTS, Q.DEFAULT_QUANT, r_info.get("quant"), m_info.get("quant"),
                    manifest.get("quant"), normalize=str.upper)
    seed = _bounded_int(m_info.get("seed"), MAX_SEED)
    if seed is None:
        seed = _bounded_int(r_info.get("seed"), MAX_SEED)
    edit_resolution = _bounded_int(m_info.get("edit_resolution"), 4096)
    resolution = _choice(RESOLUTIONS, "1K", m_info.get("resolution"), manifest.get("resolution"), normalize=str.upper)
    try:
        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
    except (ValueError, OverflowError, OSError):
        created_at = ""
    refined = _text(r_info.get("refined_prompt"))
    refine = _boolean(m_info.get("refine_prompt"), _boolean(manifest.get("refine_prompt"), bool(refined)))
    version = f"{mtime_ns}-{info.st_ctime_ns}"
    return {
        "artifact_id": artifact_id,
        "job_dir": job_dir,
        "job_id": jid,
        "filename": name,
        "url": f"/api/jobs/{artifact_id}?t={version}",
        "thumbnail_url": f"/api/gallery/thumbnail/{job_dir}/{name}?t={version}",
        "created_at": info.st_mtime,
        "created_at_str": created_at,
        "size_kb": round(info.st_size / 1024, 1),
        "width": img["width"],
        "height": img["height"],
        "has_alpha": img["has_alpha"],
        "prompt": _text(m_info.get("prompt")) or _text(r_info.get("prompt")) or "",
        "refined_prompt": refined,
        "mode": mode,
        "quant": quant,
        "seed": str(seed if seed is not None else 42),
        "refine_prompt": refine,
        "pe_variant": _choice(VARIANTS, "int8", m_info.get("pe_variant"), manifest.get("pe_variant")),
        "transparent": _boolean(m_info.get("transparent"), False),
        "aspect_ratio": _choice(Q.ASPECT_RATIOS_2K, "1:1", m_info.get("aspect_ratio"), manifest.get("aspect_ratio")),
        "resolution": RESOLUTIONS[resolution],
        "custom_size": _boolean(m_info.get("custom_size"), mode == "t2i"),
        "edit_resolution": edit_resolution if edit_resolution is not None else 1024,
        "inputs": _input_artifacts(jobs, job_dir, m_info.get("inputs")),
        "session": _text(results.get("session")),
        "gpu": _choice(GPUS, "L4", results.get("gpu"), manifest.get("_gpu"), normalize=str.upper),
        "cu_used": _number(results.get("cu_used")),
        "seconds": _number(r_info.get("seconds")),
        "sha256": _text(r_info.get("sha256")),
    }


def gallery_page(jobs: Path, limit: int, before: Position | None = None) -> tuple[list[dict], str | None]:
    if limit <= 0:
        return [], None
    metadata = {}
    items = []
    last = None
    for position in _candidates(jobs, before):
        mtime_ns, artifact_id = position
        job_dir, subfolder, filename = artifact_id.split("/")
        try:
            png = safe_subfile(jobs, job_dir, subfolder, filename)
            info = png.lstat()
        except (HTTPException, OSError):
            continue
        if info.st_mtime_ns != mtime_ns or not stat.S_ISREG(info.st_mode) or not info.st_size:
            continue
        if job_dir not in metadata:
            folder = png.parent.parent
            manifest = read_gallery_metadata(folder, "manifest.json")
            results = read_gallery_metadata(folder, "out/results.json")
            raw = manifest.get("jobs")
            m_jobs = {entry["id"]: entry for entry in raw
                      if isinstance(entry, dict) and isinstance(entry.get("id"), str)} if isinstance(raw, list) else {}
            r_jobs = results.get("jobs") if isinstance(results.get("jobs"), dict) else {}
            metadata[job_dir] = manifest, results, m_jobs, r_jobs
        item = _item(jobs, png, position, info, *metadata[job_dir])
        if item is None:
            continue
        if len(items) == limit:
            return items, f"{last[0]}:{last[1]}"
        items.append(item)
        last = position
    return items, None


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@lru_cache(maxsize=THUMBNAIL_CACHE_SIZE)
def _render_thumbnail(path: Path, identity: tuple[int, ...]) -> bytes:
    with path.open("rb") as source:
        if _identity(os.fstat(source.fileno())) != identity:
            raise OSError("Gallery image changed")
        with Image.open(source, formats=("PNG",)) as image:
            if max(image.size) > THUMBNAIL_MAX_SIDE or image.width * image.height > THUMBNAIL_MAX_PIXELS:
                raise ValueError("Gallery image exceeds thumbnail dimensions")
            mode = "RGBA" if image.mode in ("RGBA", "LA") or "transparency" in image.info else "RGB"
            with nullcontext(image) if image.mode == mode else image.convert(mode) as thumbnail, io.BytesIO() as output:
                thumbnail.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)
                thumbnail.save(output, format="PNG", compress_level=3)
                data = output.getvalue()
        if _identity(os.fstat(source.fileno())) != identity:
            raise OSError("Gallery image changed")
        return data


def gallery_thumbnail(path: Path) -> bytes:
    """Serialize cache misses to bound simultaneous full-size PNG decodes."""
    try:
        with THUMBNAIL_LOCK:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or getattr(info, "st_reparse_tag", 0):
                raise OSError("Gallery image is not a regular file")
            return _render_thumbnail(path, _identity(info))
    except (OSError, ValueError, SyntaxError, OverflowError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=404, detail="此作品的縮圖無法讀取") from exc
