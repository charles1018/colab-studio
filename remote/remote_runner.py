# -*- coding: utf-8 -*-
"""Runs ON the Colab VM. Started in the background by start.py; progress goes to /content/cstudio/status.json.

Steps:
  1. install (skipped when setup marker exists): ComfyUI + ComfyUI-GGUF + pip requirements (gguf, huggingface_hub, hf_xet)
  2. models: download required GGUF denoiser, text encoder, VAE, and optional Prompt Enhancer from Hugging Face
  3. start ComfyUI (127.0.0.1:8188), private to the VM
  4. execute every job in manifest.json -> /content/cstudio/out/<id>.png (and record refined_prompt if refine_prompt=True)
"""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cancel import CANCEL_PROTOCOL, cancel_marker  # noqa: E402
from job_paths import JOB_ID, confined_path  # noqa: E402
from models import get_model_plugin  # noqa: E402

JOB = Path(os.environ.get("CSTUDIO_DIR", "/content/cstudio"))
OUT = JOB / "out"
WORK = JOB / "work"
STATUS = JOB / "status.json"
COMFY = Path(os.environ.get("CSTUDIO_COMFY", "/content/ComfyUI"))
HF_MODELS = Path("/content/hf")
MARK = Path("/content/.cstudio_setup_v1")
COMFY_URL = os.environ.get("CSTUDIO_COMFY_URL", "http://127.0.0.1:8188")

STATE: dict = {
    "stage": "starting",
    "run_id": None,
    "cancel_protocol": CANCEL_PROTOCOL,
    "log": [],
    "jobs": {},
    "prompt_ids": [],
    "done": False,
    "error": None,
    "started": time.time(),
}

MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
MODEL_CHUNK_SIZE = 8 * 1024**2
OBJECT_INFO: dict = {}
FITTED: dict = {}
IN_FLIGHT: dict[str, bool] = {}


class BatchCancelled(Exception):
    pass


class CancellationUnconfirmed(Exception):
    pass


# ---------------------------------------------------------------- status & helpers
def save():
    tmp = confined_path(JOB, STATUS.with_suffix(".tmp").name, missing_ok=True)
    tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATUS)


def log(msg: str, stage: str | None = None):
    if stage:
        STATE["stage"] = stage
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE["log"] = (STATE["log"] + [line])[-60:]
    save()


def setup_command(args, *, check=True, read_stdout=False, capture_output=False):
    check_cancel()
    stdout = subprocess.PIPE if read_stdout or capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    with subprocess.Popen(args, stdout=stdout, stderr=stderr, text=True, start_new_session=True) as proc:
        while True:
            try:
                output, errors = proc.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if not cancellation_requested():
                    continue
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.communicate()
                raise BatchCancelled() from None
    result = subprocess.CompletedProcess(args, proc.returncode, output, errors)
    if check:
        result.check_returncode()
    return result


def cancellation_requested() -> bool:
    run_id = STATE.get("run_id")
    return bool(run_id and cancel_marker(JOB, run_id).is_file())


def check_cancel():
    if cancellation_requested():
        cancel_prompts()
        raise BatchCancelled()


def _git_head(dest: Path) -> str | None:
    r = setup_command(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def clone(url: str, dest: Path, commit: str | None = None):
    dest = Path(dest)
    if commit is None:
        if not dest.exists():
            setup_command(["git", "clone", "--depth", "1", url, str(dest)])
        else:
            setup_command(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        return
    if str(commit).startswith("-"):
        raise RuntimeError(f"bad pinned commit {commit!r}")
    if not dest.exists():
        setup_command(["git", "init", "-q", str(dest)])
        setup_command(["git", "-C", str(dest), "remote", "add", "origin", url])
    else:
        top = setup_command(["git", "-C", str(dest), "rev-parse", "--show-toplevel"], capture_output=True, check=False)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != dest.resolve():
            raise RuntimeError(f"{dest} exists but is not its own git repo")
    if setup_command(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit], check=False).returncode != 0:
        setup_command(["git", "-C", str(dest), "fetch", "-q", "origin"])
    setup_command(["git", "-C", str(dest), "checkout", "-q", "--detach", commit])
    head = _git_head(dest)
    if head != commit:
        raise RuntimeError(f"pinned checkout of {dest} failed: HEAD {head} != {commit}")


def load_pins(manifest: dict) -> dict | None:
    if manifest.get("_pin_versions") is False:
        return None
    f = JOB / "versions.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1 install
def install(pins: dict | None = None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab runtime 沒有 GPU（請用 --gpu L4 或 G4/A100）")
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"install: GPU {gpu_name}, capability {cap}, VRAM {vram_gb:.1f} GiB", "install")
    STATE["gpu_info"] = {"name": gpu_name, "capability": list(cap), "vram_gib": round(vram_gb, 2)}
    save()

    if MARK.exists():
        log("install: already done on this runtime")
        return

    pip_cmd = ["uv", "pip", "install", "--system", "-q"] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-q"]

    comfy_info = (pins or {}).get("comfyui") or {}
    comfy_repo = comfy_info.get("repo", "https://github.com/Comfy-Org/ComfyUI.git")
    comfy_commit = comfy_info.get("commit")
    log(f"install: cloning ComfyUI ({(comfy_commit or 'HEAD')[:10]})")
    clone(comfy_repo, COMFY, comfy_commit)

    nodes = (pins or {}).get("custom_nodes") or [
        {"name": "ComfyUI-GGUF", "repo": "https://github.com/city96/ComfyUI-GGUF.git", "commit": None}
    ]
    for n in nodes:
        name = n["name"]
        url = n["repo"]
        c = n.get("commit")
        log(f"install: cloning custom node {name} ({(c or 'HEAD')[:10]})")
        clone(url, COMFY / "custom_nodes" / name, c)

    log("install: installing Python packages via uv/pip")
    setup_command([*pip_cmd, "-r", str(COMFY / "requirements.txt")])
    for n in nodes:
        req = COMFY / "custom_nodes" / n["name"] / "requirements.txt"
        if req.is_file():
            setup_command([*pip_cmd, "-r", str(req)])

    extra_pip = [p.get("spec") or p["name"] for p in ((pins or {}).get("pip") or [])] or [
        "huggingface_hub>=0.26.0",
        "hf_xet",
        "gguf>=0.13.0",
    ]
    setup_command([*pip_cmd, *extra_pip])

    MARK.write_text(
        json.dumps(
            {
                "comfyui": _git_head(COMFY),
                "nodes": {n["name"]: _git_head(COMFY / "custom_nodes" / n["name"]) for n in nodes},
                "time": time.time(),
            }
        ),
        encoding="utf-8",
    )
    log("install: complete")


# ---------------------------------------------------------------- 2 models
def _model_pin_index(pins: dict | None) -> dict[tuple[str, str], dict]:
    idx: dict[tuple[str, str], dict] = {}
    if not isinstance(pins, dict):
        return idx
    for entry in pins.get("models") or []:
        if isinstance(entry, dict) and entry.get("repo") and entry.get("file"):
            idx[(entry["repo"], entry["file"])] = entry
    return idx


def _model_identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _hash_model(model, identity: list[int]) -> str:
    digest = hashlib.sha256()
    while chunk := model.read(MODEL_CHUNK_SIZE):
        check_cancel()
        digest.update(chunk)
    if _model_identity(os.fstat(model.fileno())) != identity:
        raise RuntimeError(f"模型檔案在驗證期間變更：{Path(model.name).name}")
    return digest.hexdigest()


def _model_verification_path(dest: Path) -> Path:
    return dest.with_name(f".{dest.name}.cstudio-verified.json")


def _save_model_verification(dest: Path, pin: dict, identity: list[int]):
    try:
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".cstudio-model-") as tmp:
            record = Path(tmp) / "verified.json"
            record.write_text(json.dumps({"pin": pin, "identity": identity}), encoding="utf-8")
            record.replace(_model_verification_path(dest))
    except OSError:
        log(f"models: cannot save verification for {dest.name}; will recheck next run")


def _model_cached(dest: Path, pin: dict | None) -> bool:
    info = dest.stat() if dest.is_file() else None
    if info is None:
        return False
    if pin is None:
        return info.st_size > 1024
    if info.st_size != pin["bytes"]:
        return False

    identity = _model_identity(info)
    try:
        record = json.loads(_model_verification_path(dest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if record == {"pin": pin, "identity": identity}:
        return True

    log(f"models: verifying SHA-256 of {dest.name}", "models")
    with dest.open("rb") as model:
        if _model_identity(os.fstat(model.fileno())) != identity:
            raise RuntimeError(f"模型檔案在驗證期間變更：{dest.name}")
        if _hash_model(model, identity) != pin["sha256"]:
            return False
    if _model_identity(dest.stat()) != identity:
        raise RuntimeError(f"模型檔案在驗證期間變更：{dest.name}")
    _save_model_verification(dest, pin, identity)
    return True


def _stage_model(source: Path, staged: Path, *, keep_source: bool):
    if keep_source:
        try:
            os.link(source, staged)
        except OSError:
            shutil.move(source, staged)
        return
    try:
        source.replace(staged)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        with source.open("rb") as model, staged.open("wb") as out:
            while chunk := model.read(MODEL_CHUNK_SIZE):
                check_cancel()
                out.write(chunk)


def _publish_model(staged: Path, dest: Path, pin: dict | None):
    if pin is None:
        check_cancel()
        staged.replace(dest)
        return

    with staged.open("rb") as model:
        identity = _model_identity(os.fstat(model.fileno()))
        if identity[2] != pin["bytes"]:
            raise RuntimeError(f"下載檔案大小不符 {dest.name}: 預期 {pin['bytes']} bytes, 實際 {identity[2]} bytes")
        if _hash_model(model, identity) != pin["sha256"]:
            raise RuntimeError(f"下載檔案 SHA-256 不符：{dest.name}")
        check_cancel()
        if _model_identity(staged.stat()) != identity:
            raise RuntimeError(f"模型檔案在發布前變更：{dest.name}")
        staged.replace(dest)
        # Rename can change ctime. Read the published identity from the verified open file.
        published = _model_identity(os.fstat(model.fileno()))
        if published[:-1] != identity[:-1] or _model_identity(dest.stat()) != published:
            raise RuntimeError(f"模型檔案在發布期間變更：{dest.name}")
    _save_model_verification(dest, pin, published)


def download_models(files: list[tuple[str, str, str]], pins: dict | None = None):
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    pin_idx = _model_pin_index(pins)
    # Keep the download in a process the runner can stop without interrupting ComfyUI.
    code = (
        "import json, sys\n"
        "from huggingface_hub import hf_hub_download\n"
        "repo, name, rev, local_dir = sys.argv[1:]\n"
        "path = hf_hub_download(repo_id=repo, filename=name, revision=rev or None,\n"
        "                       local_dir=local_dir, force_download=bool(rev))\n"
        "print('CSTUDIO_MODEL ' + json.dumps(path))\n"
    )

    for i, (repo, name, folder) in enumerate(files, 1):
        check_cancel()
        dest = COMFY / "models" / folder / Path(name).name
        pin = None
        if pins is not None:
            entry = pin_idx.get((repo, name), {})
            if (
                entry.get("dest") != folder
                or type(entry.get("bytes")) is not int or entry["bytes"] <= 0
                or not isinstance(entry.get("revision"), str) or not entry["revision"]
                or not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            ):
                raise RuntimeError(f"缺少有效的模型釘選資料：{repo}/{name}")
            pin = {key: entry[key] for key in ("repo", "file", "revision", "bytes", "sha256")}
            # ComfyUI searches the legacy unet directory before diffusion_models.
            legacy_unet = dest.parent.with_name("unet") / dest.name
            if folder == "diffusion_models" and legacy_unet.is_file():
                dest = legacy_unet

        if _model_cached(dest, pin):
            log(f"models [{i}/{len(files)}]: cached {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        HF_MODELS.mkdir(parents=True, exist_ok=True)
        if pin is not None:
            # The old file stays in place. Each filesystem needs room for the new file.
            needed = [(HF_MODELS, pin["bytes"] + 3 * 1024**3)]
            if HF_MODELS.stat().st_dev != dest.parent.stat().st_dev:
                needed.append((dest.parent, pin["bytes"]))
            for location, needed_bytes in needed:
                free_bytes = shutil.disk_usage(location).free
                if free_bytes < needed_bytes:
                    raise RuntimeError(
                        f"Colab 磁碟空間不足：{location} 暫存 {dest.name} 需要約 {needed_bytes / 1e9:.1f} GB，"
                        f"剩餘 {free_bytes / 1e9:.1f} GB"
                    )

        rev = pin["revision"] if pin else ""
        log(f"models [{i}/{len(files)}]: downloading {dest.name} from {repo}" + (f" @ {rev[:8]}" if rev else ""), "models")
        download_cache = (
            tempfile.TemporaryDirectory(dir=HF_MODELS, prefix=".cstudio-download-")
            if pin else nullcontext(str(HF_MODELS))
        )
        with download_cache as download_dir:
            transfer = setup_command([sys.executable, "-c", code, repo, name, rev, download_dir], read_stdout=True)
            for line in reversed(transfer.stdout.splitlines()):
                if line.startswith("CSTUDIO_MODEL "):
                    got = Path(json.loads(line.partition(" ")[2]))
                    break
            else:
                raise RuntimeError(f"模型下載程序未回傳檔案：{dest.name}")
            check_cancel()
            if got != Path(download_dir) / name or not stat.S_ISREG(got.lstat().st_mode):
                raise RuntimeError(f"模型下載程序回傳非預期檔案：{dest.name}")

            with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".cstudio-model-") as tmp:
                staged = Path(tmp) / "model.part"
                _stage_model(got, staged, keep_source=pin is None)
                _publish_model(staged, dest, pin)
        log(f"models [{i}/{len(files)}]: ready {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        check_cancel()


# ---------------------------------------------------------------- 3 ComfyUI HTTP API
class ComfyHTTPError(RuntimeError):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:600]}")
        self.code = code
        self.body = body


def call(url: str, data: dict | None = None, timeout: int = 60):
    req = (
        urllib.request.Request(url, json.dumps(data).encode("utf-8"), {"Content-Type": "application/json"})
        if data is not None
        else url
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        raise ComfyHTTPError(e.code, raw) from None


def cancel_prompts():
    if not IN_FLIGHT:
        return

    log("cancel: verifying this run's prompts", "cancelling")
    deadline = time.monotonic() + 30
    last_error = None
    while IN_FLIGHT and time.monotonic() < deadline:
        try:
            queue = call(COMFY_URL + "/queue", timeout=5)
            active = set()
            for key in ("queue_running", "queue_pending"):
                items = queue[key]
                if not isinstance(items, list):
                    raise ValueError("Invalid ComfyUI queue")
                for item in items:
                    if not isinstance(item, list) or len(item) < 2 or not isinstance(item[1], str):
                        raise ValueError("Invalid ComfyUI queue entry")
                    active.add(item[1])
            for pid, acknowledged in list(IN_FLIGHT.items()):
                if time.monotonic() >= deadline:
                    break
                if pid in active:
                    IN_FLIGHT[pid] = True
                    call(COMFY_URL + f"/api/jobs/{urllib.parse.quote(pid, safe='')}/cancel", {}, timeout=5)
                elif acknowledged:
                    del IN_FLIGHT[pid]
                else:
                    # After a lost submission reply, a prompt absent from the queue may not be visible yet.
                    history = call(COMFY_URL + f"/history/{urllib.parse.quote(pid, safe='')}", timeout=5)
                    if isinstance(history, dict) and pid in history:
                        del IN_FLIGHT[pid]
        except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = type(exc).__name__
        if IN_FLIGHT:
            time.sleep(1)
    if IN_FLIGHT:
        detail = f" ({last_error})" if last_error else ""
        raise CancellationUnconfirmed(f"無法確認 ComfyUI 已停止此批次的 prompt{detail}")


def start_comfy(has_pe: bool = False):
    check_cancel()
    try:
        call(COMFY_URL + "/system_stats", timeout=5)
        log("comfy: already running on 127.0.0.1:8188")
        return
    except Exception:
        pass
    # Keep models resident in GPU VRAM when running the 3-model base set (~14.2 GB fits inside L4 22 GiB VRAM)
    vram_gib = (STATE.get("gpu_info") or {}).get("vram_gib") or 0
    vram_flags = ["--gpu-only"] if (vram_gib >= 20 and not has_pe) else []
    log(f"comfy: starting ComfyUI server ({' '.join(vram_flags) or 'normal-vram'})", "comfy")
    subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch", *vram_flags],
        cwd=str(COMFY),
        stdout=open("/content/comfyui.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(90):
        check_cancel()
        time.sleep(2)
        try:
            call(COMFY_URL + "/system_stats", timeout=5)
            log("comfy: ready")
            return
        except Exception:
            pass
    tail = Path("/content/comfyui.log").read_text(errors="ignore")[-2000:]
    raise RuntimeError("ComfyUI did not start within 180s:\n" + tail)


def combo_options(spec):
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    first = spec[0]
    if isinstance(first, list):
        return first
    if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict) and isinstance(spec[1].get("options"), list):
        return spec[1]["options"]
    return None


def fit_combo_names(g: dict, object_info: dict):
    changes, missing = [], []
    for nid, node in g.items():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type")
        if ctype not in object_info:
            missing.append(f"node {nid} ({ctype}): 節點未載入或未安裝")
            continue
        info = object_info.get(ctype, {}).get("input", {})
        specs = {}
        for part in ("required", "optional"):
            if isinstance(info.get(part), dict):
                specs.update(info[part])
        for key, val in (node.get("inputs") or {}).items():
            if not isinstance(val, str) or not val.lower().endswith(MODEL_EXT):
                continue
            opts = combo_options(specs.get(key))
            if opts is None or val in opts:
                continue
            want = val.replace("\\", "/")
            hit = [o for o in opts if isinstance(o, str) and o.replace("\\", "/") == want]
            if not hit:
                name = want.rsplit("/", 1)[-1]
                hit = [o for o in opts if isinstance(o, str) and o.replace("\\", "/").rsplit("/", 1)[-1] == name]
            if len(hit) == 1:
                node["inputs"][key] = hit[0]
                changes.append((nid, key, val, hit[0]))
            else:
                missing.append(f"{nid}.{key}: {val}")
    return changes, missing


def _object_info(refresh: bool = False) -> dict | None:
    if refresh:
        OBJECT_INFO.clear()
    if not OBJECT_INFO:
        try:
            info = call(COMFY_URL + "/object_info", timeout=120)
            if isinstance(info, dict):
                OBJECT_INFO.update(info)
        except Exception as e:
            log(f"could not read /object_info: {e}")
            return None
    return OBJECT_INFO


def fit_names(graph: dict):
    info = _object_info()
    if info is None:
        return []
    original = copy.deepcopy(graph)
    changes, missing = fit_combo_names(graph, info)
    if missing:
        info = _object_info(refresh=True)
        if info is not None:
            fresh = copy.deepcopy(original)
            changes, missing = fit_combo_names(fresh, info)
            graph.clear()
            graph.update(fresh)
    for nid, key, old, new in changes:
        log(f"name fitted: {nid}.{key}: {old} -> {new}")
    if missing:
        raise RuntimeError("ComfyUI 缺少模型或節點: " + "; ".join(missing[:6]))
    return changes


def queue_prompt(graph: dict, *, pin_models: bool = False) -> str:
    fitted = fit_names(graph)
    if pin_models and fitted:
        raise RuntimeError("ComfyUI 模型路徑偏離已驗證的檔名：" + ", ".join(new for _, _, _, new in fitted))
    check_cancel()
    pid = str(uuid.uuid4())
    IN_FLIGHT[pid] = False
    STATE["prompt_ids"].append(pid)
    save()
    try:
        resp = call(COMFY_URL + "/prompt", {"prompt": graph, "client_id": "cstudio", "prompt_id": pid})
    except ComfyHTTPError as exc:
        if exc.code == 400:
            IN_FLIGHT.pop(pid)
        raise
    if not isinstance(resp, dict) or not resp.get("prompt_id"):
        raise RuntimeError(f"ComfyUI 回應沒有 prompt_id：{json.dumps(resp, ensure_ascii=False)[:300]}")
    active_pid = resp["prompt_id"]
    if active_pid != pid:
        IN_FLIGHT.pop(pid)
        STATE["prompt_ids"][-1] = active_pid
    IN_FLIGHT[active_pid] = True
    if resp.get("node_errors"):
        raise RuntimeError(f"ComfyUI node_errors：{json.dumps(resp['node_errors'], ensure_ascii=False)[:400]}")
    if fitted:
        FITTED[active_pid] = fitted
    save()
    return active_pid


def wait_history(pid: str, timeout: int, what: str) -> dict:
    t0, last = time.time(), time.time()
    fails = 0
    while time.time() - t0 < timeout:
        cancelling = cancellation_requested()
        if cancelling:
            cancel_prompts()
        try:
            h = call(COMFY_URL + f"/history/{pid}").get(pid)
            fails = 0
        except Exception as e:
            if cancelling:
                raise BatchCancelled() from e
            fails += 1
            if fails > 5:
                raise RuntimeError(f"{what}: 輪詢 ComfyUI 連續失敗 {fails} 次：{e}")
            time.sleep(2)
            continue

        if h and h.get("status", {}).get("completed") is not None:
            IN_FLIGHT.pop(pid, None)
            if cancelling and not h["status"].get("completed"):
                raise BatchCancelled()
            if h["status"].get("status_str") == "error":
                msgs = [m[1] for m in h["status"].get("messages", []) if m[0] == "execution_error"]
                m = msgs[0] if msgs else {}
                raise RuntimeError(
                    f"ComfyUI 執行錯誤: {m.get('node_type', '')} {m.get('exception_message', 'error')[:500]}".strip()
                )
            return h
        if cancelling:
            raise BatchCancelled()
        if time.time() - last > 30:
            last = time.time()
            log(f"{what}: generating ({time.time() - t0:.0f}s)")
        time.sleep(1.5)
    raise RuntimeError(f"{what}: 超過 {timeout} 秒未完成")


def stage_input_image(rel_path: str, job_id: str, idx: int) -> str:
    """Copy an uploaded input image from /content/cstudio/inputs/<rel_path> into ComfyUI/input/."""
    input_root = confined_path(JOB, "inputs", directory=True)
    src = confined_path(input_root, rel_path)
    ext = src.suffix.lower() or ".png"
    safe_name = f"cstudio_{job_id}_in{idx}{ext}"
    dest = confined_path(COMFY, f"input/{safe_name}", missing_ok=True)
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(src, dest)
    return safe_name


def extract_refined_prompt(history_entry: dict) -> str | None:
    """Extract enhanced prompt text from node '33' (PreviewAny) or '32' (TextGenerate) in history outputs."""
    outputs = history_entry.get("outputs") or {}
    for nid in ("33", "32"):
        out = outputs.get(nid)
        if isinstance(out, dict):
            txt = out.get("text")
            if isinstance(txt, list) and txt:
                val = str(txt[0]).strip()
                if val and val != "None":
                    return val
            elif isinstance(txt, str) and txt.strip() and txt.strip() != "None":
                return txt.strip()
    return None


def collect_output_image(history_entry: dict, dest_path: Path, *, save_node: str, prefix: str) -> Path:
    """Copy only the PNG produced by the submitted SaveImage node."""
    outputs = history_entry.get("outputs")
    saved = outputs.get(save_node) if isinstance(outputs, dict) else None
    imgs = saved.get("images") if isinstance(saved, dict) else None
    if not isinstance(imgs, list) or not imgs or not isinstance(imgs[0], dict):
        raise RuntimeError("ComfyUI 完成但未在指定 SaveImage 節點找到輸出圖片")
    info = imgs[0]
    fn = info.get("filename")
    if (
        info.get("type") != "output" or info.get("subfolder") != ""
        or not isinstance(fn, str) or not re.fullmatch(rf"{re.escape(prefix)}_[0-9]+_\.png", fn)
    ):
        raise RuntimeError("ComfyUI 輸出路徑與提交的 SaveImage 設定不符")
    candidate = confined_path(COMFY, f"output/{fn}", missing_ok=True)
    if candidate.is_file():
        shutil.copy2(candidate, dest_path)
    else:
        qs = urllib.parse.urlencode({"filename": fn, "subfolder": "", "type": "output"})
        with urllib.request.urlopen(f"{COMFY_URL}/view?{qs}", timeout=60) as r:
            dest_path.write_bytes(r.read())
    return dest_path


# ---------------------------------------------------------------- 4 main batch execution
def run_batch():
    manifest_path = confined_path(JOB, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    STATE["run_id"] = manifest.get("_run_id")
    save()
    check_cancel()
    jobs = manifest.get("jobs") or []
    for job in jobs:
        jid = job["id"]
        if not isinstance(jid, str) or not JOB_ID.fullmatch(jid):
            raise ValueError("遠端 manifest 含有不合法的任務 ID")

    pins = load_pins(manifest)
    pin_models = manifest.get("_pin_versions") is not False and bool(manifest.get("_pin_models", True))
    if pin_models and not isinstance(pins, dict):
        raise RuntimeError("缺少有效的模型釘選表 versions.json")
    plugin = get_model_plugin(manifest.get("model"))
    cli_quant = manifest.get("_cli_quant")

    # 1. Install ComfyUI & ComfyUI-GGUF
    install(pins)
    check_cancel()

    # 2. Download required models for this batch
    req_models = plugin.required_models_for_batch(manifest, cli_quant=cli_quant)
    download_models(req_models, pins=pins if pin_models else None)
    check_cancel()

    # 3. Start ComfyUI
    has_pe = any(bool(j.get("refine_prompt", manifest.get("refine_prompt", False))) for j in (manifest.get("jobs") or []))
    start_comfy(has_pe=has_pe)

    confined_path(JOB, OUT.name, directory=True, missing_ok=True).mkdir(exist_ok=True)
    confined_path(JOB, WORK.name, directory=True, missing_ok=True).mkdir(exist_ok=True)

    log(f"batch: starting {len(jobs)} job(s)", "running")

    for idx, job in enumerate(jobs, 1):
        check_cancel()
        jid = job["id"]
        t_job = time.time()
        STATE["jobs"][jid] = {"status": "running", "started": t_job}
        save()
        log(f"job [{idx}/{len(jobs)}] {jid}: preparing ({job.get('mode', 't2i')})")

        try:
            staged_inputs: list[str] = []
            for i_in, rel_in in enumerate(job.get("inputs") or [], 1):
                staged_inputs.append(stage_input_image(rel_in, jid, i_in))

            graph = plugin.build_qwen21_graph(
                job,
                manifest=manifest,
                cli_quant=cli_quant,
                uploaded_inputs=staged_inputs,
            )
            pid = queue_prompt(graph, pin_models=pin_models)
            refine_on = bool(job.get("refine_prompt", manifest.get("refine_prompt", False)))
            timeout_s = int(job.get("timeout", 1200 if refine_on else 600))
            h = wait_history(pid, timeout=timeout_s, what=jid)

            refined_text = None
            if refine_on:
                refined_text = extract_refined_prompt(h)
                if not refined_text:
                    raise RuntimeError(
                        f"{jid}: refine_prompt=True 但 Prompt Enhancer 輸出為空（可能 thinking 耗盡 token 上限，請調高 pe_max_length 或設 thinking=false）"
                    )
                log(f"job {jid}: prompt enhanced ({len(refined_text)} chars)")

            out_file = confined_path(JOB, f"{OUT.name}/{jid}.png", missing_ok=True)
            save_node = next(nid for nid, node in graph.items() if node["class_type"] == "SaveImage")
            collect_output_image(h, out_file, save_node=save_node, prefix=graph[save_node]["inputs"]["filename_prefix"])
            sha256 = hashlib.sha256(out_file.read_bytes()).hexdigest()
            elapsed = round(time.time() - t_job, 2)

            job_rec = {
                "status": "done",
                "file": f"out/{jid}.png",
                "bytes": out_file.stat().st_size,
                "sha256": sha256,
                "seconds": elapsed,
                "seed": int(job.get("seed", 42)),
                "quant": (cli_quant or job.get("quant") or manifest.get("quant") or plugin.DEFAULT_QUANT).upper(),
                "mode": job.get("mode", "t2i"),
            }
            if refined_text is not None:
                job_rec["refined_prompt"] = refined_text
            STATE["jobs"][jid] = job_rec
            save()
            log(f"job [{idx}/{len(jobs)}] {jid}: done in {elapsed:.1f}s -> {out_file.name}")
        except (BatchCancelled, CancellationUnconfirmed):
            raise
        except Exception as e:
            tb = traceback.format_exc()
            STATE["jobs"][jid] = {
                "status": "error",
                "error": str(e),
                "traceback": tb[-1500:],
                "seconds": round(time.time() - t_job, 2),
            }
            save()
            log(f"job [{idx}/{len(jobs)}] {jid}: FAILED - {e}")

    bad = [k for k, v in STATE["jobs"].items() if v.get("status") != "done"]
    if bad or IN_FLIGHT:
        check_cancel()
    STATE["done"] = True
    STATE["stage"] = "done" if not bad else "done_with_errors"
    if bad:
        STATE["error"] = f"{len(bad)}/{len(jobs)} jobs failed: {', '.join(bad)}"
    save()
    log(f"batch finished: {len(jobs) - len(bad)}/{len(jobs)} succeeded")


def main():
    try:
        run_batch()
    except BatchCancelled:
        STATE.update(done=True, stage="cancelled", error=None)
        log("batch cancelled")
    except CancellationUnconfirmed as exc:
        STATE.update(done=True, stage="cancel_failed", error=str(exc))
        log(f"cancel unconfirmed: {exc}")
    except Exception as exc:
        STATE["done"] = True
        STATE["stage"] = "error"
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log(f"FATAL: {STATE['error']}")
        traceback.print_exc()
        save()
        sys.exit(1)


if __name__ == "__main__":
    main()
