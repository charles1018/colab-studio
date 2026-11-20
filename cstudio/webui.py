# -*- coding: utf-8 -*-
"""FastAPI backend and StudioUIState manager for Colab Studio local Web UI."""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from remote.job_paths import confined_path, relative_parts

from . import colab_client as C
from . import colab_login as L
from . import runner as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC_DIR = HERE / "static"
JOBS_DIR = ROOT / "jobs"
REMOTE = ROOT / "remote"
if str(REMOTE) not in sys.path:
    sys.path.insert(0, str(REMOTE))
from models import qwen_image_21 as Q  # noqa: E402

STAGE_LABELS_ZH: dict[str, str] = {
    "idle": "待命中",
    "booting_vm": "🚀 正在 Google Colab 分配並啟動 GPU...",
    "reusing_vm": "♻️ 正在連線沿用開啟中的 Colab GPU...",
    "syncing_code": "📦 正在同步遠端引擎與任務設定...",
    "uploading_inputs": "⬆️ 正在上傳參考圖片至 Colab...",
    "starting": "▶️ 正在啟動遠端生成引擎...",
    "install": "🔧 正在安裝雲端 ComfyUI 與自訂節點環境...",
    "models": "⬇️ 正在高速下載／驗證 HuggingFace 權重模型...",
    "comfy": "⚙️ 正在啟動雲端 ComfyUI 伺服器...",
    "generate": "🎨 正在執行 Qwen-Image-2.1 8 步極速取樣（含 Prompt Enhancer 擴寫）...",
    "syncing_outputs": "⬇️ 遠端生成完成，正在下載回本地並執行 SHA256 校驗...",
    "done": "✅ 生成與 SHA256 校驗下載完成！",
    "cancelling": "⏳ 已要求取消，等待遠端確認",
    "cancelled": "⏹️ 已確認任務取消",
    "cancel_failed": "⚠️ 無法確認遠端任務已停止",
    "stopping": "🛑 正在關閉 Colab GPU",
    "stopped": "✅ 已確認 Colab Session 關閉",
    "stopped_local": "⏹️ 已停止本地批次",
    "shutdown_pending": "⏳ Colab Session 關機待確認",
    "error": "❌ 執行發生錯誤",
    "crashed": "❌ 遠端程序異常中止",
}


def get_quant_catalog() -> list[dict[str, Any]]:
    """Build catalog of all 14 supported quantization specs (2 Native + 12 Unsloth GGUF) with GB sizes."""
    versions = R.load_versions()
    pin_idx = {(e["repo"], e["file"]): e for e in (versions.get("models") or []) if "repo" in e and "file" in e}
    items: list[dict[str, Any]] = []

    # Put default INT8 and BF16 first, followed by popular GGUFs
    ordered_quants = ["INT8", "BF16"] + list(Q.GGUF_QUANTS)
    descriptions = {
        "INT8": "原生 INT8 ConvRot（官方預設推薦，速度與畫質最佳平衡）",
        "BF16": "原生 BF16 全精度（適合 A100 / H100 / L4）",
        "Q8_0": "Unsloth GGUF 8-bit（極高畫質，適合 L4 / G4）",
        "Q6_K_XL": "Unsloth GGUF 6-bit XL（高細節保留）",
        "Q6_K": "Unsloth GGUF 6-bit（高畫質推薦）",
        "Q5_K_M": "Unsloth GGUF 5-bit Medium（均衡推薦）",
        "Q5_K_S": "Unsloth GGUF 5-bit Small",
        "Q4_K_M": "Unsloth GGUF 4-bit Medium（輕量快速，L4 熱門選擇）",
        "Q4_K_S": "Unsloth GGUF 4-bit Small",
        "Q3_K_XL": "Unsloth GGUF 3-bit XL",
        "Q3_K_M": "Unsloth GGUF 3-bit Medium",
        "Q3_K_S": "Unsloth GGUF 3-bit Small",
        "Q2_K": "Unsloth GGUF 2-bit（極限輕量測試用）",
        "F16": "Unsloth GGUF FP16 全精度",
    }
    for q in ordered_quants:
        if q in Q.NATIVE_QUANTS:
            repo, fn, _ = Q.NATIVE_QUANTS[q]
            kind = "native"
        else:
            repo = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
            fn = Q.gguf_filename(q)
            kind = "gguf"
        entry = pin_idx.get((repo, fn)) or {}
        b = int(entry.get("bytes") or 0)
        gb = round(b / 1e9, 2) if b else None
        desc = descriptions.get(q, f"Unsloth GGUF {q}")
        label = f"{q} ({gb:.1f} GB) — {desc}" if gb else f"{q} — {desc}"
        items.append(
            {
                "id": q,
                "kind": kind,
                "file": fn.rsplit("/", 1)[-1],
                "bytes": b,
                "gb": gb,
                "description": desc,
                "label": label,
                "is_default": q == Q.DEFAULT_QUANT,
            }
        )
    return items


def read_local_keeps() -> list[dict[str, Any]]:
    """Read local ~/.config/colab-studio/keep/*.json without spawning colab CLI."""
    now = time.time()
    account = C.account_scope()
    out: list[dict[str, Any]] = []
    if not C.KEEP.is_dir():
        return out
    for p in sorted(C.KEEP.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                continue
            sess = str(d.get("session") or p.stem)
            exp = d.get("expires")
            owner = d.get("created_by", "external")
            release_pending = bool(d.get("release_pending"))
            rem_sec = max(0, int(round(exp - now))) if isinstance(exp, (int, float)) else None
            if (not account or d.get("account_scope", account) != account
                    or (owner == "owned" and not d.get("runtime_endpoint"))):
                owner, exp, rem_sec, release_pending = "unverified", None, None, False
            out.append(
                {
                    "session": sess,
                    "owner": owner,
                    "expires": exp,
                    "remaining_sec": rem_sec,
                    "busy": exp is None and owner not in ("pending", "unverified"),
                    "release_pending": release_pending,
                    "watchdog_alive": C.watchdog_alive(sess),
                }
            )
        except Exception:
            continue
    return out


def safe_job_dir(job_dir: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", job_dir) or job_dir in (".", ".."):
        raise HTTPException(status_code=400, detail=f"不合法的任務資料夾名稱：{job_dir!r}")
    try:
        return confined_path(JOBS_DIR, job_dir, directory=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="找不到任務資料夾") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="不允許存取此任務資料夾") from exc


def safe_subfile(job_dir: str, subfolder: str, filename: str) -> Path:
    if subfolder not in ("out", "inputs"):
        raise HTTPException(status_code=400, detail="不支援的子資料夾")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", filename) or filename in (".", ".."):
        raise HTTPException(status_code=400, detail=f"不合法的檔案名稱：{filename!r}")
    if subfolder == "out" and not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="輸出作品必須是 PNG")
    base = safe_job_dir(job_dir)
    try:
        return confined_path(base, f"{subfolder}/{filename}")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"找不到圖片檔案：{job_dir}/{subfolder}/{filename}") from exc


def inspect_image_meta(img_path: Path) -> dict[str, Any]:
    try:
        with Image.open(img_path) as im:
            has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in im.info)
            return {"width": im.width, "height": im.height, "mode": im.mode, "has_alpha": has_alpha}
    except Exception:
        return {"width": None, "height": None, "mode": None, "has_alpha": False}


def read_gallery_metadata(folder: Path, relative: str) -> dict[str, Any]:
    try:
        data = json.loads(confined_path(folder, relative).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def scan_gallery(limit: int = 120) -> list[dict[str, Any]]:
    """Scan local jobs/*/out/results.json and manifest.json to build rich gallery items."""
    if JOBS_DIR.is_symlink() or JOBS_DIR.is_junction() or not JOBS_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for candidate in JOBS_DIR.iterdir():
        job_dir_name = candidate.name
        try:
            folder = safe_job_dir(job_dir_name)
            out_dir = confined_path(folder, "out", directory=True)
        except (HTTPException, OSError, ValueError):
            continue
        manifest_data = read_gallery_metadata(folder, "manifest.json")
        results_data = read_gallery_metadata(folder, "out/results.json")

        m_jobs_by_id: dict[str, dict[str, Any]] = {}
        raw_m_jobs = manifest_data.get("jobs")
        for j in (raw_m_jobs if isinstance(raw_m_jobs, list) else []):
            if isinstance(j, dict) and isinstance(j.get("id"), str):
                m_jobs_by_id[j["id"]] = j

        r_jobs = results_data.get("jobs") if isinstance(results_data.get("jobs"), dict) else {}
        # Collect PNG files in out/
        for candidate_png in out_dir.glob("*.png"):
            if candidate_png.name.startswith(".") or candidate_png.name.endswith("_sheet.png"):
                continue
            try:
                png = safe_subfile(job_dir_name, "out", candidate_png.name)
                info = png.stat()
            except (HTTPException, OSError):
                continue
            if info.st_size == 0:
                continue
            jid = png.stem
            # If manifest has jobs defined and jid is not one of them (e.g. a user-created collage), skip
            if m_jobs_by_id and jid not in m_jobs_by_id and jid not in r_jobs:
                continue
            r_info = r_jobs.get(jid) if isinstance(r_jobs.get(jid), dict) else {}
            m_info = m_jobs_by_id.get(jid) or {}

            # If results.json exists and explicitly marks this job as non-done error, skip
            if r_info and r_info.get("status") not in (None, "done"):
                continue

            img_meta = inspect_image_meta(png)
            mode = str(m_info.get("mode") or r_info.get("mode") or "t2i").lower()
            raw_inputs = m_info.get("inputs") if isinstance(m_info.get("inputs"), list) else []
            input_urls: list[dict[str, str]] = []
            for rel_in in raw_inputs:
                try:
                    parts = relative_parts(rel_in)
                    if len(parts) != 2 or parts[0] != "inputs":
                        continue
                    fn = safe_subfile(job_dir_name, *parts).name
                except (HTTPException, ValueError):
                    continue
                input_urls.append(
                    {
                        "filename": fn,
                        "artifact_id": f"{job_dir_name}/inputs/{fn}",
                        "url": f"/api/jobs/{job_dir_name}/inputs/{fn}",
                    }
                )

            seed_val = m_info.get("seed", r_info.get("seed", 42))
            quant_val = str(
                r_info.get("quant")
                or m_info.get("quant")
                or manifest_data.get("quant")
                or Q.DEFAULT_QUANT
            ).upper()
            refine_val = bool(
                m_info.get("refine_prompt", manifest_data.get("refine_prompt", bool(r_info.get("refined_prompt"))))
            )
            pe_var_val = str(m_info.get("pe_variant") or manifest_data.get("pe_variant") or "int8").lower()
            transparent_val = bool(m_info.get("transparent", False))
            ar_val = str(m_info.get("aspect_ratio") or manifest_data.get("aspect_ratio") or "1:1")
            res_val = str(m_info.get("resolution") or manifest_data.get("resolution") or "1K").upper()
            custom_size_val = bool(m_info.get("custom_size", mode == "t2i"))
            edit_res_val = int(m_info.get("edit_resolution", 1024))

            mtime = info.st_mtime
            items.append(
                {
                    "artifact_id": f"{job_dir_name}/out/{png.name}",
                    "job_dir": job_dir_name,
                    "job_id": jid,
                    "filename": png.name,
                    "url": f"/api/jobs/{job_dir_name}/out/{png.name}?t={int(mtime)}",
                    "created_at": mtime,
                    "created_at_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                    "size_kb": round(info.st_size / 1024.0, 1),
                    "width": img_meta["width"] or r_info.get("width"),
                    "height": img_meta["height"] or r_info.get("height"),
                    "has_alpha": img_meta["has_alpha"],
                    "prompt": str(m_info.get("prompt") or r_info.get("prompt") or ""),
                    "refined_prompt": r_info.get("refined_prompt"),
                    "mode": mode,
                    "quant": quant_val,
                    "seed": str(seed_val),
                    "refine_prompt": refine_val,
                    "pe_variant": pe_var_val,
                    "transparent": transparent_val,
                    "aspect_ratio": ar_val,
                    "resolution": res_val,
                    "custom_size": custom_size_val,
                    "edit_resolution": edit_res_val,
                    "inputs": input_urls,
                    "session": results_data.get("session"),
                    "gpu": results_data.get("gpu") or manifest_data.get("_gpu") or "L4",
                    "cu_used": results_data.get("cu_used"),
                    "seconds": r_info.get("seconds"),
                    "sha256": r_info.get("sha256"),
                }
            )

    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items[:limit]


def resolve_artifact_to_path(artifact_id: str) -> Path:
    """Safely resolve a server-issued artifact_id ('<job_dir>/out/<file>' or '<job_dir>/inputs/<file>') to a local file."""
    try:
        parts = relative_parts(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="不合法的圖片識別碼") from exc
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail=f"不合法的圖片識別碼：{artifact_id!r}")
    job_dir, subfolder, filename = parts
    return safe_subfile(job_dir, subfolder, filename)


class StudioUIState:
    """Thread-safe state manager for Colab Studio Web UI."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.colab_query_lock = threading.Lock()
        self.revision: int = 0
        self.running: bool = False
        self.provisioning: bool = False
        self.cancel_requested: bool = False
        self.abort_action: str | None = None
        self.run_id: str | None = None
        self.job_dir: str | None = None
        self.job_id: str | None = None
        self.stage: str = "idle"
        self.stage_label: str = STAGE_LABELS_ZH["idle"]
        self.session: str | None = None
        self.session_account: str | None = None
        self.shutdown_endpoint: str | None = None
        self.gpu: str = "L4"
        self.quant: str = Q.DEFAULT_QUANT
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.logs: list[str] = []
        self.error: str | None = None
        self.synced_images: list[dict[str, Any]] = []
        self.last_recipe: dict[str, Any] | None = None

        # Cached slow-path Colab status
        self.colab_cache_account: str | None = None
        self.colab_cache: dict[str, Any] = {
            "checked_at": 0.0,
            "token_ok": C.token_ok(),
            "cu": None,
            "cu_raw": "尚未查詢",
            "sessions": [],
            "sessions_raw": "",
            "refreshing": False,
        }

    def _bump(self) -> int:
        self.revision += 1
        return self.revision

    def append_log(self, msg: str) -> None:
        with self.lock:
            if not self.logs or self.logs[-1] != msg:
                self.logs.append(msg)
                if len(self.logs) > 80:
                    self.logs = self.logs[-80:]
                self._bump()

    def on_runner_progress(self, st: dict[str, Any]) -> None:
        with self.lock:
            if st.get("run_id"):
                self.run_id = str(st["run_id"])
            if st.get("session"):
                self.session = str(st["session"])
            if st.get("gpu"):
                self.gpu = str(st["gpu"])
            raw_stage = str(st.get("stage") or self.stage or "starting")
            if raw_stage == "booting_vm":
                self.provisioning = True
            elif raw_stage in ("reusing_vm", "syncing_code"):
                self.provisioning = False
            if self.abort_action != "stop":
                self.stage = raw_stage
                self.stage_label = STAGE_LABELS_ZH.get(raw_stage, f"🔄 執行中 ({raw_stage})...")

            for line in st.get("log") or []:
                s_line = str(line).strip()
                if s_line and s_line not in self.logs:
                    self.logs.append(s_line)
            if len(self.logs) > 80:
                self.logs = self.logs[-80:]

            if st.get("error") and raw_stage in ("error", "crashed", "cancelled"):
                self.error = str(st["error"])

            synced_ids = st.get("_local_synced_jobs") or []
            local_res = st.get("_local_results") or {}
            r_jobs = local_res.get("jobs") if isinstance(local_res.get("jobs"), dict) else {}
            if self.job_dir and synced_ids:
                new_synced: list[dict[str, Any]] = []
                for jid in synced_ids:
                    try:
                        png = safe_subfile(self.job_dir, "out", f"{jid}.png")
                        info = png.stat()
                    except (HTTPException, OSError):
                        continue
                    if info.st_size > 0:
                        jrec = r_jobs.get(jid) if isinstance(r_jobs.get(jid), dict) else {}
                        mtime = info.st_mtime
                        img_meta = inspect_image_meta(png)
                        new_synced.append(
                            {
                                "artifact_id": f"{self.job_dir}/out/{png.name}",
                                "job_dir": self.job_dir,
                                "job_id": jid,
                                "filename": png.name,
                                "url": f"/api/jobs/{self.job_dir}/out/{png.name}?t={int(mtime)}",
                                "refined_prompt": jrec.get("refined_prompt"),
                                "width": img_meta["width"] or jrec.get("width"),
                                "height": img_meta["height"] or jrec.get("height"),
                                "has_alpha": img_meta["has_alpha"],
                                "seconds": jrec.get("seconds"),
                                "sha256": jrec.get("sha256"),
                            }
                        )
                self.synced_images = new_synced

            self._bump()

    def fast_status(self) -> dict[str, Any]:
        now = time.time()
        keeps = read_local_keeps()
        with self.lock:
            elapsed = round(now - self.started_at, 1) if (self.running and self.started_at) else (
                round(self.finished_at - self.started_at, 1)
                if (self.started_at and self.finished_at)
                else 0.0
            )
            active_sess = self.session
            if not active_sess and keeps:
                active_sess = keeps[0]["session"]
            active_keep = next((k for k in keeps if k["session"] == active_sess), None)
            return {
                "revision": self.revision,
                "running": self.running,
                "cancel_requested": self.cancel_requested,
                "run_id": self.run_id,
                "job_dir": self.job_dir,
                "job_id": self.job_id,
                "stage": self.stage,
                "stage_label": self.stage_label,
                "session": active_sess,
                "gpu": self.gpu,
                "quant": self.quant,
                "elapsed_sec": elapsed,
                "logs": list(self.logs[-30:]),
                "error": self.error,
                "synced_images": list(self.synced_images),
                "last_recipe": self.last_recipe,
                "local_keeps": keeps,
                "active_keep": active_keep,
                "token_ok": C.token_ok(),
            }

    def refresh_colab_cache(self, force: bool = False) -> dict[str, Any]:
        account = C.account_scope()
        now = time.time()
        with self.lock:
            age = now - float(self.colab_cache.get("checked_at") or 0.0)
            if not force and account == self.colab_cache_account and self.colab_cache.get("checked_at") and age < 30.0:
                return dict(self.colab_cache, local_keeps=read_local_keeps())

        # Single-flight query so concurrent requests don't spawn duplicate `colab` CLI processes
        with self.colab_query_lock:
            account = C.account_scope()
            now2 = time.time()
            with self.lock:
                age2 = now2 - float(self.colab_cache.get("checked_at") or 0.0)
                if not force and account == self.colab_cache_account and self.colab_cache.get("checked_at") and age2 < 10.0:
                    return dict(self.colab_cache, local_keeps=read_local_keeps())
                self.colab_cache["refreshing"] = True

            tok = C.token_ok()
            cu_val: float | None = None
            cu_raw = "尚未登入"
            sess_names: list[str] = []
            sess_raw = ""
            if account:
                cu_val, cu_raw = C.usage(account=account)
                try:
                    sess_names, sess_raw = C.sessions(check=True, account=account)
                except (OSError, RuntimeError, SystemExit):
                    sess_raw = "無法確認 Colab Session 清單"

            keeps_by_name = {k["session"]: k for k in read_local_keeps()}
            assignments = C.listed_assignments(sess_raw)
            endpoints = {name: ep for ep, name in (assignments or {}).items()}
            sessions_detail: list[dict[str, Any]] = []
            seen_names: set[str] = set()
            for n in sess_names:
                seen_names.add(n)
                k = keeps_by_name.get(n)
                record = C.keep_info(n) or {}
                if record.get("runtime_endpoint") != endpoints.get(n) or record.get("account_scope", account) != account:
                    k = None
                sessions_detail.append(
                    {
                        "name": n,
                        "alive": True,
                        "owned": C.is_owned_session(n, account=account, assignments=assignments),
                        "remaining_sec": k["remaining_sec"] if k else None,
                        "busy": k["busy"] if k else False,
                        "release_pending": k["release_pending"] if k else False,
                    }
                )
            for k_name, k in keeps_by_name.items():
                if k_name not in seen_names:
                    sessions_detail.append(
                        {
                            "name": k_name,
                            "alive": False,
                            "owned": k["owner"] == "owned",
                            "remaining_sec": k["remaining_sec"],
                            "busy": k["busy"],
                            "release_pending": k["release_pending"],
                        }
                    )

            updated = {
                "checked_at": time.time(),
                "checked_at_str": time.strftime("%H:%M:%S"),
                "token_ok": tok,
                "cu": cu_val,
                "cu_raw": cu_raw,
                "sessions": sessions_detail,
                "active_session_names": sess_names,
                "sessions_raw": sess_raw,
                "refreshing": False,
                "local_keeps": list(keeps_by_name.values()),
            }
            with self.lock:
                self.colab_cache = updated
                self.colab_cache_account = account
                if (
                    not self.running and self.session and self.stage == "shutdown_pending"
                    and account and account == self.session_account == C.account_scope()
                    and not C.keep_file(self.session).exists()
                    and C.session_gone(self.session, sess_names, sess_raw, self.shutdown_endpoint)
                ):
                    self.session = None
                    self.shutdown_endpoint = None
                    self.stage = "stopped"
                    self.stage_label = STAGE_LABELS_ZH["stopped"]
                self._bump()
            return dict(updated)


STATE = StudioUIState()
app = FastAPI(title="Colab Studio Local Web UI", version="0.1.0")
LOCAL_UI_HOSTS = ("127.0.0.1", "localhost", "::1")
LOCAL_AUTHORITY = re.compile(r"(127\.0\.0\.1|localhost|\[::1\])(?::([0-9]{1,5}))?", re.IGNORECASE | re.ASCII)
UI_SESSION_TOKEN = secrets.token_urlsafe(32)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def is_loopback_peer(host: str) -> bool:
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def require_local_ui_access(request: Request, call_next):
    headers = request.headers
    if not request.client or not is_loopback_peer(request.client.host) or any(
        name in ("forwarded", "x-real-ip") or name.startswith("x-forwarded-") for name in headers
    ):
        return JSONResponse({"detail": "工作台僅接受直接的本機 loopback 連線"}, status_code=403)

    hosts = headers.getlist("host")
    authority = LOCAL_AUTHORITY.fullmatch(hosts[0]) if len(hosts) == 1 else None
    scheme = request.scope["scheme"]
    default_port = 443 if scheme == "https" else 80
    if not authority:
        return JSONResponse({"detail": "不允許此工作台 Host"}, status_code=400)
    port = int(authority[2]) if authority[2] else default_port
    if not 1 <= port <= 65535:
        return JSONResponse({"detail": "不允許此工作台 Host"}, status_code=400)

    expected_origin = f"{scheme}://{authority[1].lower()}"
    if port != default_port:
        expected_origin += f":{port}"
    origins = headers.getlist("origin")
    if origins and origins != [expected_origin]:
        return JSONResponse({"detail": "不允許跨來源存取工作台"}, status_code=403)

    path = request.scope["path"]
    root_path = request.scope.get("root_path", "")
    if root_path and path.startswith(f"{root_path}/"):
        path = path[len(root_path):]
    safe_method = request.method in ("GET", "HEAD")
    public = safe_method and (path == "/" or path.startswith("/static/"))
    cookie_name = f"cstudio_ui_{port}"
    if not public:
        fetch_sites = headers.getlist("sec-fetch-site")
        if fetch_sites and fetch_sites not in (["same-origin"], ["none"]):
            return JSONResponse({"detail": "不允許跨來源存取工作台"}, status_code=403)
        cookie = request.cookies.get(cookie_name, "")
        if not secrets.compare_digest(cookie.encode(), UI_SESSION_TOKEN.encode()):
            return JSONResponse({"detail": "請先開啟 Colab Studio 首頁"}, status_code=401)
        if not safe_method and not origins:
            return JSONResponse({"detail": "控制請求需要同源 Origin"}, status_code=403)
        # Cross-origin browser requests with this header require a CORS preflight.
        # Require it for slow status too, since that GET contacts Colab.
        if (not safe_method or path == "/api/colab/status") and headers.getlist("x-colab-studio-request") != ["1"]:
            return JSONResponse({"detail": "缺少工作台請求標頭"}, status_code=403)

    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    if not public or path == "/":
        response.headers["Cache-Control"] = "no-store"
    if path == "/" and request.method == "GET" and response.status_code == 200:
        response.set_cookie(cookie_name, UI_SESSION_TOKEN, httponly=True, samesite="strict", secure=scheme == "https")
    return response


def parse_bool_form(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


@app.get("/", response_class=HTMLResponse)
def index_page():
    idx = STATIC_DIR / "index.html"
    if not idx.is_file():
        return HTMLResponse("<h1>找不到 cstudio/static/index.html</h1>", status_code=500)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


@app.get("/api/meta")
def api_meta():
    """Return static model capabilities, aspect ratios, GPUs, and 14 quant options."""
    ar_list = []
    for ar, (w2, h2) in Q.ASPECT_RATIOS_2K.items():
        w1, h1 = Q.ASPECT_RATIOS_1K[ar]
        ar_list.append(
            {
                "id": ar,
                "w_1k": w1,
                "h_1k": h1,
                "w_2k": w2,
                "h_2k": h2,
            }
        )
    return {
        "gpus": list(C.GPUS),
        "default_gpu": "L4",
        "default_quant": Q.DEFAULT_QUANT,
        "quants": get_quant_catalog(),
        "aspect_ratios": ar_list,
        "pe_variants": [
            {"id": "int8", "label": "Qwen 3.5 9B INT8（官方預設高精度擴寫）"},
            {"id": "w4a8", "label": "Qwen 3.5 9B W4A8（輕量快速擴寫）"},
        ],
    }


@app.get("/api/colab/status")
def api_colab_status(force: bool = Query(False)):
    data = STATE.refresh_colab_cache(force=force)
    return data


@app.post("/api/colab/login/url")
def api_login_url():
    try:
        url, state = L.get_login_url()
        return {"ok": True, "url": url, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"產生 Google OAuth 登入網址失敗：{e}") from e


@app.post("/api/colab/login/code")
def api_login_code(payload: dict[str, Any]):
    code = str(payload.get("code") or "").strip()
    state = payload.get("state")
    if not code:
        raise HTTPException(status_code=400, detail="請貼上 Google 授權碼")
    try:
        tok_path = L.submit_login_code(code, state=str(state) if state else None)
        colab_st = STATE.refresh_colab_cache(force=True)
        return {"ok": True, "token_path": str(tok_path), "colab": colab_st}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google OAuth 授權碼驗證失敗：{e}") from e


def _resolve_target_session(requested: str | None) -> str:
    req = str(requested or "").strip()
    if req:
        try:
            return C.sanitize_session_name(req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    account = C.account_scope()
    with STATE.lock:
        if STATE.session and STATE.session_account in (None, account):
            try:
                return C.sanitize_session_name(STATE.session)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
    keeps = [keep for keep in read_local_keeps() if keep["owner"] not in ("pending", "unverified")]
    if len(keeps) == 1:
        return C.sanitize_session_name(keeps[0]["session"])
    names, _ = C.sessions()
    owned = [n for n in names if C.is_owned_session(n)]
    if len(owned) == 1:
        return owned[0]
    if len(names) == 1:
        return names[0]
    if not names:
        raise HTTPException(status_code=400, detail="目前沒有開啟中的 Colab Session")
    raise HTTPException(status_code=400, detail=f"有多個開啟中的 Session，請先選擇目標 Session：{', '.join(names)}")


@app.post("/api/colab/keep")
def api_colab_keep(payload: dict[str, Any]):
    try:
        minutes = float(payload.get("minutes") or 30.0)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="保留分鐘數 (minutes) 必須為數字") from e
    minutes = max(1.0, min(240.0, minutes))
    account = C.account_scope()
    session = _resolve_target_session(payload.get("session"))
    now = time.time()
    record = C.keep_info(session)
    cur_info = record or {}
    if cur_info.get("release_pending"):
        raise HTTPException(status_code=409, detail="此 Session 正在關機，無法延長待命")
    try:
        target = C.keep_target(session, cur_info, account)
        cur_exp = cur_info.get("expires")
        base_time = max(now, float(cur_exp)) if isinstance(cur_exp, (int, float)) else now
        new_exp = min(now + 240.0 * 60.0, base_time + minutes * 60.0)
        C.set_keep(session, new_exp, expected=record, **target)
        C.start_watchdog(session)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    rem_sec = max(0, int(round(new_exp - now)))
    STATE.append_log(f"⏸️ 已延長 {session} 自動關機時間（目前剩餘約 {round(rem_sec / 60, 1)} 分鐘）")
    return {
        "ok": True,
        "session": session,
        "minutes": minutes,
        "remaining_sec": rem_sec,
        "local_keeps": read_local_keeps(),
    }


@app.post("/api/colab/cancel")
def api_colab_cancel(payload: dict[str, Any] | None = None):
    payload = payload or {}
    requested = str(payload.get("session") or "").strip()
    run_id = str(payload["run_id"]).strip() if payload.get("run_id") else None
    if run_id and not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=400, detail="不合法的批次 ID")
    if requested:
        requested = _resolve_target_session(requested)
    with STATE.lock:
        if STATE.running and (not requested or requested == STATE.session):
            if STATE.session_account is not None and STATE.session_account != C.account_scope():
                raise HTTPException(status_code=409, detail="工作台的 Colab 登入身分已變更")
            if run_id and run_id != STATE.run_id:
                raise HTTPException(status_code=409, detail="工作台的執行中批次已變更")
            if STATE.abort_action == "stop":
                raise HTTPException(status_code=409, detail="此 Session 正在關機")
            STATE.cancel_requested = True
            STATE.abort_action = "cancel"
            STATE.stage = "cancelling"
            STATE.stage_label = STAGE_LABELS_ZH["cancelling"]
            STATE._bump()
            return JSONResponse(status_code=202, content={
                "ok": True, "state": "pending", "session": STATE.session, "run_id": STATE.run_id,
            })

    target_session = _resolve_target_session(requested)
    try:
        res = C.request_cancel(target_session, run_id)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="未收到可確認的遠端取消結果") from exc
    state = res["state"]
    if state == "unsupported":
        raise HTTPException(status_code=409, detail="此遠端批次使用舊版取消協定")
    if state == "failed":
        raise HTTPException(status_code=409, detail="遠端未能確認指定批次取消")
    return JSONResponse(
        status_code=202 if state == "pending" else 200,
        content={"ok": True, "session": target_session, **res},
    )


@app.post("/api/colab/stop")
def api_colab_stop(payload: dict[str, Any] | None = None):
    payload = payload or {}
    stop_all = payload.get("all", False)
    requested = payload.get("session")
    if type(stop_all) is not bool or (requested is not None and not isinstance(requested, str)):
        raise HTTPException(status_code=400, detail="all 必須為布林值，session 必須為字串")
    if stop_all and requested is not None:
        raise HTTPException(status_code=400, detail="請擇一指定 all 或 session")
    if requested is not None:
        try:
            requested = C.sanitize_session_name(requested)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    account = C.account_scope()
    owned = set(C.owned_sessions()) if requested is None else set()
    with STATE.lock:
        active = STATE.session if STATE.running else None
        if STATE.session_account is not None and STATE.session_account != account:
            if requested is not None and requested == STATE.session:
                raise HTTPException(status_code=409, detail="工作台的 Colab 登入身分已變更")
            active = None
        if active and STATE.provisioning:
            owned.add(active)
        if stop_all:
            targets = sorted(owned)
        elif requested is not None:
            targets = [requested]
        elif active in owned:
            targets = [active]
        elif len(owned) == 1:
            targets = sorted(owned)
        else:
            raise HTTPException(status_code=400, detail="請指定要關閉的 Colab Session")
        deferred = None
        if active in targets and (requested is not None or STATE.provisioning or C.is_owned_session(active)):
            STATE.cancel_requested = True
            STATE.abort_action = "stop"
            STATE.stage = "stopping"
            STATE.stage_label = STAGE_LABELS_ZH["stopping"]
            STATE._bump()
            if STATE.provisioning and (C.keep_info(active) or {}).get("created_by") != "owned":
                deferred = active

    stopped = []
    pending = []
    failed = []
    for session in targets:
        if session == deferred:
            pending.append(session)
            continue
        outcome = (C.release_runtime(session, owned_only=requested is None, account=account)
                   if account else C.ReleaseState.FAILED)
        if outcome == C.ReleaseState.CONFIRMED:
            stopped.append(session)
        elif outcome == C.ReleaseState.PENDING:
            pending.append(session)
        else:
            failed.append(session)
    with STATE.lock:
        if not STATE.running and STATE.session in stopped:
            STATE.session = None
            STATE.stage = "stopped"
            STATE.stage_label = STAGE_LABELS_ZH["stopped"]
        elif not STATE.running and STATE.session in pending:
            STATE.session_account = account
            STATE.shutdown_endpoint = (C.keep_info(STATE.session) or {}).get("runtime_endpoint")
            STATE.stage = "shutdown_pending"
            STATE.stage_label = STAGE_LABELS_ZH["shutdown_pending"]
        STATE._bump()
    result = {
        "ok": not pending and not failed, "requested": targets, "stopped": stopped,
        "verified_gone": stopped, "pending": pending, "failed": failed,
    }
    if failed and not (stopped or pending):
        result["detail"] = f"無法確認關機：{', '.join(failed)}"
        return JSONResponse(status_code=502, content=result)
    return JSONResponse(status_code=207 if failed else 202 if pending else 200, content=result)


@app.get("/api/run/status")
def api_run_status():
    return STATE.fast_status()


@app.get("/api/gallery")
def api_gallery(limit: int = Query(120, ge=1, le=500)):
    return {"items": scan_gallery(limit=limit)}


@app.get("/api/jobs/{job_dir}/{subfolder}/{filename}")
def api_serve_job_file(job_dir: str, subfolder: str, filename: str):
    target = safe_subfile(job_dir, subfolder, filename)
    return FileResponse(target, media_type="image/png", filename=filename)


@app.post("/api/run/start")
async def api_run_start(
    mode: str = Form("t2i"),
    prompt: str = Form(""),
    refine_prompt: str = Form("false"),
    pe_variant: str = Form("int8"),
    transparent: str = Form("false"),
    aspect_ratio: str = Form("1:1"),
    resolution: str = Form("1K"),
    edit_sizing: str = Form("match_1k"),
    seed: str = Form(""),
    gpu: str = Form("L4"),
    quant: str = Form("INT8"),
    session: str = Form(""),
    idle_minutes: float = Form(15.0),
    stop_after: str = Form("false"),
    batch_name: str = Form(""),
    ref_order: str = Form("[]"),
    ref_files: list[UploadFile] = File(default=[]),
):
    """Validate inputs, create local jobs/webui_<timestamp>/ folder, and start background runner."""
    with STATE.lock:
        if STATE.running:
            raise HTTPException(
                status_code=409,
                detail="目前已有正在執行的生成任務，請先等待完成或點擊「⏹️ 中斷當前任務」",
            )

    if not C.token_ok():
        raise HTTPException(
            status_code=401,
            detail="尚未登入 Google Colab OAuth，請先點擊頂部狀態列的「🔑 登入 Google Colab」",
        )
    account = C.account_scope()
    if not account:
        raise HTTPException(status_code=401, detail="無法確認 Colab 登入身分，請重新登入")

    mode_clean = str(mode or "t2i").strip().lower()
    if mode_clean not in ("t2i", "edit"):
        raise HTTPException(status_code=400, detail=f"不支援的生圖模式：{mode_clean!r}")

    prompt_clean = str(prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="請輸入提示詞 (Prompt)")

    gpu_clean = str(gpu or "L4").strip().upper()
    if gpu_clean not in C.GPUS:
        raise HTTPException(status_code=400, detail=f"不支援的 GPU：{gpu_clean!r}")

    quant_clean = str(quant or Q.DEFAULT_QUANT).strip().upper()
    if quant_clean not in Q.VALID_QUANTS:
        raise HTTPException(status_code=400, detail=f"不支援的權重規格：{quant_clean!r}")

    pe_var_clean = str(pe_variant or "int8").strip().lower()
    if pe_var_clean not in ("int8", "w4a8"):
        raise HTTPException(status_code=400, detail=f"不支援的 Prompt Enhancer 規格：{pe_var_clean!r}")

    ar_clean = str(aspect_ratio or "1:1").strip()
    if ar_clean not in Q.ASPECT_RATIOS_2K:
        raise HTTPException(status_code=400, detail=f"不支援的長寬比：{ar_clean!r}")

    res_clean = str(resolution or "1K").strip().upper()
    if res_clean not in ("1K", "2K"):
        raise HTTPException(status_code=400, detail=f"不支援的解析度：{res_clean!r}")

    refine_bool = parse_bool_form(refine_prompt, False)
    transparent_bool = parse_bool_form(transparent, False)
    stop_after_bool = parse_bool_form(stop_after, False)
    idle_mins = max(1.0, min(240.0, float(idle_minutes or 15.0)))

    seed_str = str(seed or "").strip()
    if seed_str:
        try:
            seed_int = int(seed_str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"種子碼 (Seed) 必須為整數：{seed_str!r}") from e
        if not (0 <= seed_int <= 2**63 - 1):
            raise HTTPException(status_code=400, detail=f"種子碼 (Seed) 必須為 0 ～ {2**63 - 1} 之間的正整數")
    else:
        seed_int = random.randint(1, 2_147_483_647)

    # Build safe local job directory name
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    custom_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(batch_name or "").strip()).strip("_")[:32]
    folder_name = f"webui_{ts_str}_{custom_slug}_{short_uid}" if custom_slug else f"webui_{ts_str}_{short_uid}"

    resolved_sources: list[tuple[str, Any]] = []
    if mode_clean == "edit":
        try:
            order_spec = json.loads(ref_order or "[]")
            if not isinstance(order_spec, list):
                order_spec = []
        except Exception:
            order_spec = []

        # Map uploaded files by their client-side key or index
        upload_idx = 0
        if order_spec:
            for item in order_spec:
                if not isinstance(item, dict):
                    continue
                kind = item.get("kind")
                if kind == "upload" and upload_idx < len(ref_files):
                    resolved_sources.append(("upload", ref_files[upload_idx]))
                    upload_idx += 1
                elif kind == "artifact" and item.get("artifact_id"):
                    src_path = resolve_artifact_to_path(item["artifact_id"])
                    resolved_sources.append(("artifact", src_path))
        else:
            for uf in ref_files:
                resolved_sources.append(("upload", uf))

        if not (1 <= len(resolved_sources) <= 10):
            raise HTTPException(
                status_code=400,
                detail=f"圖片編輯（Edit）模式需要 1～10 張參考圖片（目前收到 {len(resolved_sources)} 張）",
            )

    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        job_folder = confined_path(JOBS_DIR, folder_name, directory=True, missing_ok=True)
        job_folder.mkdir()
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="任務資料夾已經存在") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="無法使用任務資料夾") from exc
    inputs_folder = job_folder / "inputs"
    out_folder = job_folder / "out"
    inputs_folder.mkdir()
    out_folder.mkdir()

    saved_inputs_rel: list[str] = []
    if mode_clean == "edit":
        for idx, (src_kind, src_obj) in enumerate(resolved_sources, start=1):
            dst_name = f"ref_{idx}.png"
            dst_path = inputs_folder / dst_name
            try:
                if src_kind == "upload":
                    raw_bytes = await src_obj.read()
                    if len(raw_bytes) > 35 * 1024 * 1024:
                        raise ValueError(f"參考圖片 <image{idx}> 超過 35MB 上限")
                    with Image.open(io.BytesIO(raw_bytes)) as im:
                        im.load()
                        norm_im = im.convert("RGBA") if (im.mode in ("RGBA", "LA") or "transparency" in im.info) else im.convert("RGB")
                        norm_im.save(dst_path, format="PNG")
                else:
                    with Image.open(src_obj) as im:
                        im.load()
                        norm_im = im.convert("RGBA") if (im.mode in ("RGBA", "LA") or "transparency" in im.info) else im.convert("RGB")
                        norm_im.save(dst_path, format="PNG")
            except Exception as e:
                shutil.rmtree(job_folder, ignore_errors=True)
                raise HTTPException(status_code=400, detail=f"參考圖片 <image{idx}> 讀取或轉檔失敗：{e}") from e
            saved_inputs_rel.append(f"inputs/{dst_name}")

    # Build job spec
    job_id = custom_slug if (custom_slug and re.fullmatch(r"[A-Za-z0-9_-]{1,60}", custom_slug)) else f"img_{ts_str}"
    job_spec: dict[str, Any] = {
        "id": job_id,
        "mode": mode_clean,
        "prompt": prompt_clean,
        "quant": quant_clean,
        "seed": seed_int,
        "refine_prompt": refine_bool,
        "pe_variant": pe_var_clean,
        "transparent": transparent_bool,
    }

    if mode_clean == "t2i":
        job_spec["aspect_ratio"] = ar_clean
        job_spec["resolution"] = res_clean
    else:
        job_spec["inputs"] = saved_inputs_rel
        sizing = str(edit_sizing or "match_1k").strip().lower()
        if sizing == "match_2k":
            job_spec["edit_resolution"] = 2048
            job_spec["custom_size"] = False
        elif sizing == "match_orig":
            job_spec["edit_resolution"] = 0
            job_spec["custom_size"] = False
        elif sizing == "custom":
            job_spec["edit_resolution"] = 2048 if res_clean == "2K" else 1024
            job_spec["custom_size"] = True
            job_spec["aspect_ratio"] = ar_clean
            job_spec["resolution"] = res_clean
        else:
            job_spec["edit_resolution"] = 1024
            job_spec["custom_size"] = False

    manifest_data: dict[str, Any] = {
        "model": "qwen-image-2.1-turbo",
        "quant": quant_clean,
        "_gpu": gpu_clean,
        "jobs": [job_spec],
    }

    problems = R.validate_manifest(manifest_data, job_folder)
    if problems:
        shutil.rmtree(job_folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="任務設定驗證未通過：\n- " + "\n- ".join(problems))

    manifest_path = job_folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # The runner selects automatic reuse from the authenticated assignment listing.
    req_sess = str(session or "").strip()
    target_session = None
    if req_sess and req_sess != "__new__":
        try:
            target_session = C.sanitize_session_name(req_sess)
        except ValueError as e:
            shutil.rmtree(job_folder, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e)) from e

    with STATE.lock:
        if STATE.running:
            raise HTTPException(status_code=409, detail="另一個生成任務剛啟動，請稍候")
        STATE.running = True
        STATE.provisioning = False
        STATE.cancel_requested = False
        STATE.abort_action = None
        STATE.run_id = None
        STATE.job_dir = folder_name
        STATE.job_id = job_id
        STATE.stage = "booting_vm"
        STATE.stage_label = f"🚀 正在確認 Google Colab {gpu_clean} 的目標 Session..."
        STATE.session = target_session
        STATE.session_account = account
        STATE.shutdown_endpoint = None
        STATE.gpu = gpu_clean
        STATE.quant = quant_clean
        STATE.started_at = time.time()
        STATE.finished_at = None
        STATE.error = None
        STATE.synced_images = []
        STATE.logs = [
            f"📋 已建立本地任務：{folder_name} (mode={mode_clean}, quant={quant_clean}, seed={seed_int})",
            f"🎯 目標 Colab Session：{target_session or '待確認'} (GPU={gpu_clean})",
        ]
        STATE.last_recipe = {
            "job_dir": folder_name,
            "job_id": job_id,
            "mode": mode_clean,
            "prompt": prompt_clean,
            "quant": quant_clean,
            "seed": str(seed_int),
            "refine_prompt": refine_bool,
            "pe_variant": pe_var_clean,
            "transparent": transparent_bool,
            "aspect_ratio": ar_clean,
            "resolution": res_clean,
            "edit_sizing": edit_sizing,
            "gpu": gpu_clean,
            "session": target_session,
        }
        STATE._bump()

    def _worker():
        try:
            res = R.run_manifest(
                manifest_path,
                gpu=gpu_clean,
                session=target_session,
                account=account,
                auto_reuse=not req_sess,
                reuse_only=target_session is not None,
                cli_quant=quant_clean,
                idle_minutes=idle_mins,
                stop_after=stop_after_bool,
                progress_cb=STATE.on_runner_progress,
                exit_on_error=False,
                should_abort=lambda: (STATE.abort_action or ("cancel" if STATE.cancel_requested else None)),
            )
            with STATE.lock:
                if res.get("shutdown_state") == C.ReleaseState.PENDING:
                    STATE.stage = "shutdown_pending"
                    STATE.stage_label = STAGE_LABELS_ZH["shutdown_pending"]
                elif res.get("error"):
                    STATE.stage = "error"
                    STATE.stage_label = STAGE_LABELS_ZH["error"]
                    STATE.error = str(res["error"])
                else:
                    STATE.stage = "done"
                    STATE.stage_label = STAGE_LABELS_ZH["done"]
                    if res.get("shutdown_state") == C.ReleaseState.CONFIRMED:
                        STATE.session = None
                STATE._bump()
        except BaseException as exc:
            with STATE.lock:
                if isinstance(exc, R.RunCancelled):
                    STATE.stage = "cancelled"
                    STATE.stage_label = STAGE_LABELS_ZH["cancelled"]
                    STATE.error = None
                elif isinstance(exc, R.RunStopped) and exc.outcome != C.ReleaseState.FAILED:
                    if exc.outcome == C.ReleaseState.CONFIRMED:
                        STATE.stage = "stopped"
                        STATE.session = None
                    else:
                        STATE.stage = "shutdown_pending" if exc.outcome == C.ReleaseState.PENDING else "stopped_local"
                    STATE.stage_label = STAGE_LABELS_ZH[STATE.stage]
                    STATE.error = None
                else:
                    STATE.stage = "error"
                    STATE.stage_label = STAGE_LABELS_ZH["error"]
                    STATE.error = str(exc)
                STATE.logs.append(f"⚠️ {exc}")
                STATE._bump()
        finally:
            with STATE.lock:
                if STATE.stage == "shutdown_pending" and STATE.session:
                    STATE.shutdown_endpoint = (C.keep_info(STATE.session) or {}).get("runtime_endpoint")
                STATE.running = False
                STATE.provisioning = False
                STATE.finished_at = time.time()
                STATE._bump()
            try:
                STATE.refresh_colab_cache(force=True)
            except Exception:
                pass

    t = threading.Thread(target=_worker, name=f"cstudio-run-{folder_name}", daemon=True)
    t.start()

    return JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "job_dir": folder_name,
            "job_id": job_id,
            "session": target_session,
            "seed": str(seed_int),
            "status": STATE.fast_status(),
        },
    )


def launch_ui(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = False):
    import uvicorn

    host = host.lower()
    if host not in LOCAL_UI_HOSTS:
        raise SystemExit("工作台只允許 --host 127.0.0.1、localhost 或 ::1")
    bind_host = "127.0.0.1" if host == "localhost" else host
    url_host = "[::1]" if host == "::1" else host
    url = f"http://{url_host}:{port}"
    print(f"🎨 啟動 Colab Studio 本地輕量 Web UI：{url}", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=bind_host, port=port, log_level="info", proxy_headers=False)


def main():
    ap = argparse.ArgumentParser(prog="cstudio-ui", description="啟動 Colab Studio 本地輕量 Web UI")
    ap.add_argument("--host", default="127.0.0.1", help="loopback 監聽位址 (127.0.0.1、localhost 或 ::1)")
    ap.add_argument("--port", type=int, default=7860, help="監聽埠號 (預設 7860)")
    ap.add_argument("--open", action="store_true", help="啟動後自動開啟瀏覽器")
    args = ap.parse_args()
    launch_ui(host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
