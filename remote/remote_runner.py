# -*- coding: utf-8 -*-
"""Runs on the Colab VM with a batch-specific workspace and shared model cache.

Steps:
  1. verify or install ComfyUI, ComfyUI-GGUF and their Python requirements
  2. models: download required GGUF denoiser, text encoder, VAE, and optional Prompt Enhancer from Hugging Face
  3. start ComfyUI (127.0.0.1:8188), private to the VM
  4. execute the batch manifest and write results and status in its run directory
"""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import select
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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cancel import CANCEL_PROTOCOL, cancel_marker  # noqa: E402
from job_paths import JOB_ID, RUN_ID, SHA256, confined_path, file_sha256  # noqa: E402
from manifest import normalize_manifest, normalize_quant, parse_manifest  # noqa: E402
from models import get_model_plugin  # noqa: E402

JOB = Path(os.environ.get("CSTUDIO_DIR", "/content/cstudio"))
RUNTIME = Path(os.environ.get("CSTUDIO_RUNTIME_DIR", str(JOB)))
MANAGED_RUN_ID = os.environ.get("CSTUDIO_RUN_ID")
OUT = JOB / "out"
WORK = JOB / "work"
STATUS = JOB / "status.json"
COMFY = Path(os.environ.get("CSTUDIO_COMFY", "/content/ComfyUI"))
HF_MODELS = Path("/content/hf")
MARK = Path("/content/.cstudio_setup_v1")
COMFY_PROCESS = RUNTIME / "comfy-process.json"
COMFY_LOG = Path("/content/comfyui.log")
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
COMFY_URL = os.environ.get("CSTUDIO_COMFY_URL", DEFAULT_COMFY_URL)
COMFY_INSTANCE = "CSTUDIO_COMFY_INSTANCE"
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF.git"
EXTRA_PIP = ("huggingface_hub>=0.26.0", "hf_xet", "gguf>=0.13.0")

STATE: dict = {
    "stage": "starting",
    "run_id": MANAGED_RUN_ID,
    "bundle_hash": os.environ.get("CSTUDIO_BUNDLE_HASH"),
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


def _installed_checkout(dest: Path, url: str) -> str | None:
    if dest.is_symlink():
        raise RuntimeError(f"Git checkout 使用了符號連結：{dest}")
    if not dest.exists():
        return None
    git = ["git", "-C", str(dest)]
    top = setup_command([*git, "rev-parse", "--show-toplevel"], capture_output=True, check=False)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != dest.resolve():
        raise RuntimeError(f"無法驗證 Git checkout：{dest}")
    origin = setup_command([*git, "remote", "get-url", "origin"], capture_output=True, check=False)
    if origin.returncode != 0 or origin.stdout.strip() != url:
        raise RuntimeError(f"Git origin 與預期不符：{dest}")
    status = setup_command([*git, "status", "--porcelain", "--untracked-files=all"], capture_output=True)
    if status.stdout.strip():
        raise RuntimeError(f"Git checkout 有未提交的變更：{dest}")
    return _git_head(dest)


def clone(url: str, dest: Path, commit: str | None = None) -> str:
    if not dest.exists() and commit is None:
        setup_command(["git", "clone", "--depth", "1", "--", url, str(dest)])
    else:
        git = ["git", "-C", str(dest)]
        if not dest.exists():
            setup_command(["git", "init", "-q", str(dest)])
            setup_command([*git, "remote", "add", "origin", url])
        revision = commit or "HEAD"
        if commit is None or setup_command(
            [*git, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True, check=False,
        ).returncode:
            fetched = setup_command(
                [*git, "fetch", "-q", "--depth", "1", "origin", revision], capture_output=True, check=False,
            )
            if fetched.returncode == 0:
                revision = "FETCH_HEAD"
            elif commit is None:
                fetched.check_returncode()
            else:
                setup_command([*git, "fetch", "-q", "origin"])
        setup_command([*git, "checkout", "-q", "--detach", revision])
    head = _installed_checkout(dest, url)
    if head is None or (commit is not None and head != commit):
        raise RuntimeError(f"Git checkout 未取得預期版本：{dest}")
    return head


def load_pins(manifest: dict) -> dict | None:
    if manifest.get("_pin_versions") is False:
        return None
    try:
        versions_dir = Path(__file__).resolve().parent if MANAGED_RUN_ID else JOB
        pins = json.loads(confined_path(versions_dir, "versions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("缺少有效的釘選表 versions.json") from exc
    if not isinstance(pins, dict):
        raise RuntimeError("缺少有效的釘選表 versions.json")
    return pins


def _read_record(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _write_record(path: Path, record: dict):
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".cstudio-state-") as tmp:
        staged = Path(tmp) / "record.json"
        staged.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        staged.replace(path)


# ---------------------------------------------------------------- 1 install
def _install_plan(pins: dict | None) -> tuple[list[tuple[Path, str, str | None]], list[str]]:
    nodes_dir = COMFY / "custom_nodes"
    if pins is None:
        return [(COMFY, COMFY_REPO, None), (nodes_dir / "ComfyUI-GGUF", GGUF_REPO, None)], list(EXTRA_PIP)

    try:
        comfy = pins["comfyui"]
        nodes = pins["custom_nodes"]
        repos = [(COMFY, comfy["repo"], comfy["commit"])]
        for node in nodes:
            name = node["name"]
            if not JOB_ID.fullmatch(name):
                raise ValueError("invalid custom node name")
            repos.append((nodes_dir / name, node["repo"], node["commit"]))
        if not nodes or len({path for path, _, _ in repos}) != len(repos):
            raise ValueError("missing or duplicate custom nodes")
        for _, url, commit in repos:
            if not isinstance(url, str) or not url or url.startswith("-") or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("invalid Git pin")
        extra = []
        for package in pins["pip"]:
            spec = package.get("spec") or package["name"]
            if not isinstance(spec, str) or not spec.strip() or spec.lstrip().startswith("-"):
                raise ValueError("invalid package spec")
            extra.append(spec)
            if package.get("version") is not None:
                name = package["name"]
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
                    raise ValueError("invalid package name")
                extra.append(f"{name}=={package['version']}")
        if not extra:
            raise ValueError("missing package specs")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("versions.json 含有不完整的 ComfyUI、節點或 Python 釘選資料") from exc
    return repos, extra


def _verify_custom_nodes(managed: set[Path], comfy_head: str | None):
    nodes_dir = COMFY / "custom_nodes"
    git = ["git", "-C", str(COMFY)]
    tracked = {}
    if comfy_head is not None:
        tree = setup_command([*git, "ls-tree", "-z", comfy_head, "--", "custom_nodes/"], read_stdout=True)
        for entry in tree.stdout.split("\0"):
            metadata, _, relative = entry.partition("\t")
            mode, _, blob = metadata.partition(" blob ")
            if mode in ("100644", "100755"):
                tracked[relative] = blob

    for path in nodes_dir.glob("*"):
        if path in managed or path.name.endswith(".disabled") or path.name == "__pycache__":
            continue
        if path.suffix != ".py" and not any((path / name).is_file() for name in ("__init__.py", "prestartup_script.py")):
            continue
        relative = path.relative_to(COMFY).as_posix()
        if relative in tracked and not path.is_symlink() and path.is_file():
            # Compare the worktree bytes to the commit, even if Git status skips the file.
            actual = setup_command([*git, "hash-object", "--no-filters", "--", relative], read_stdout=True)
            if actual.stdout.strip() == tracked[relative]:
                continue
        raise RuntimeError(f"釘選模式發現未列入 versions.json 的 ComfyUI 節點：{path.name}")


def _requirements_versions(requirements: list[Path], extra: list[str]) -> dict[str, str] | None:
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        return None

    specs = list(extra)
    for path in requirements:
        specs.extend(path.read_text(encoding="utf-8").splitlines())
    versions = {}
    for spec in specs:
        spec = spec.split(" #", 1)[0].strip()
        if not spec or spec.startswith("#"):
            continue
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate():
            continue
        name = canonicalize_name(req.name)
        try:
            installed = version(name)
        except PackageNotFoundError:
            return None
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            return None
        versions[name] = installed
    return versions


def install(pins: dict | None = None) -> str:
    repos, extra = _install_plan(pins)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab runtime 沒有 GPU（請用 --gpu L4 或 G4/A100）")
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"install: GPU {gpu_name}, capability {cap}, VRAM {vram_gb:.1f} GiB", "install")
    STATE["gpu_info"] = {"name": gpu_name, "capability": list(cap), "vram_gib": round(vram_gb, 2)}
    save()

    heads = {str(path): _installed_checkout(path, url) for path, url, _ in repos}
    managed_nodes = {path for path, _, _ in repos[1:]}
    if pins is not None:
        _verify_custom_nodes(managed_nodes, heads[str(COMFY)])
    updates = [
        (path, url, commit) for path, url, commit in repos
        if heads[str(path)] is None or (commit is not None and heads[str(path)] != commit)
    ]
    if updates:
        _stop_comfy_for_install()
        for path, url, commit in updates:
            log(f"install: updating {path.name} ({(commit or 'HEAD')[:10]})")
            heads[str(path)] = clone(url, path, commit)
        if pins is not None:
            _verify_custom_nodes(managed_nodes, heads[str(COMFY)])
    requirements = [path / "requirements.txt" for path, _, _ in repos]
    requirements = [path for path in requirements if path.is_file()]
    packages = _requirements_versions(requirements, extra)
    if packages is None:
        if not updates:
            _stop_comfy_for_install()
        log("install: installing Python packages via uv/pip")
        pip_cmd = ["uv", "pip", "install", "--system", "-q"] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-q"]
        args = [arg for path in requirements for arg in ("-r", str(path))]
        setup_command([*pip_cmd, *args, *extra])
        packages = _requirements_versions(requirements, extra)
        if packages is None:
            raise RuntimeError("安裝後的 Python 依賴仍未符合 versions.json 與 requirements.txt")

    environment = {
        "python": [sys.executable, sys.version], "heads": heads, "packages": packages,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }
    signature = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    STATE["environment"] = environment
    record = {"schema": 2, "environment": environment, "signature": signature}
    if _read_record(MARK) == record:
        log("install: verified cached environment")
    else:
        _write_record(MARK, record)
        log("install: verified current environment")
    return signature


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


def _prepared_model(repo: str, name: str, dest: Path, pin: dict | None) -> dict:
    info = dest.stat()
    record = {"repo": repo, "file": name, "path": dest.relative_to(COMFY).as_posix(), "bytes": info.st_size}
    if pin is None:
        record["mtime_ns"] = info.st_mtime_ns
    else:
        record.update(revision=pin["revision"], sha256=pin["sha256"])
    return record


def download_models(files: list[tuple[str, str, str]], pins: dict | None = None) -> dict:
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

    prepared = {}
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
            record = prepared[(repo, name, folder)] = _prepared_model(repo, name, dest, pin)
            log(f"models [{i}/{len(files)}]: cached {dest.name} ({record['bytes'] / 1e9:.2f} GB)")
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
        record = prepared[(repo, name, folder)] = _prepared_model(repo, name, dest, pin)
        log(f"models [{i}/{len(files)}]: ready {dest.name} ({record['bytes'] / 1e9:.2f} GB)")
        check_cancel()
    return prepared


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


def _comfy_system() -> dict | None:
    try:
        stats = call(COMFY_URL + "/system_stats", timeout=5)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return None
        raise
    system = stats.get("system") if isinstance(stats, dict) else None
    argv = system.get("argv") if isinstance(system, dict) else None
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
        raise RuntimeError("無法確認正在執行的 ComfyUI 啟動參數")
    versions = {
        key: system[key] for key in ("comfyui_version", "python_version", "pytorch_version")
        if isinstance(system.get(key), str) and system[key]
    }
    return {"args": argv, "versions": versions}


def _owns_comfy_process(record: dict) -> bool:
    pid, token = record.get("pid"), record.get("token")
    if type(pid) is not int or pid <= 0 or not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
        return False
    try:
        env = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return f"{COMFY_INSTANCE}={token}".encode() in env


def _comfy_loopback_port() -> int | None:
    url = urllib.parse.urlsplit(COMFY_URL)
    if url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost"):
        return url.port or 80
    return None


def _comfy_listener_sockets() -> set[str]:
    port = _comfy_loopback_port()
    if port is None:
        return set()
    address = f"0100007F:{port:04X}"
    rows = Path("/proc/net/tcp").read_text(encoding="ascii").splitlines()[1:]
    return {
        f"socket:[{row[9]}]" for line in rows
        if len(row := line.split()) > 9 and row[1] == address and row[3] == "0A"
    }


def _process_has_listener(proc: Path, sockets: set[str]) -> bool:
    try:
        for fd in (proc / "fd").iterdir():
            try:
                if str(fd.readlink()) in sockets:
                    return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def _legacy_comfy_checkout() -> bool:
    sockets = _comfy_listener_sockets()
    root = COMFY.resolve()
    python = Path(sys.executable).resolve()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (
                (proc / "cwd").resolve(strict=True) == root
                and (proc / "exe").resolve(strict=True) == python
                and _process_has_listener(proc, sockets)
            ):
                return True
        except OSError:
            continue
    return False


def _stop_owned_comfy(record: dict):
    proc = Path("/proc") / str(record["pid"])
    if not _process_has_listener(proc, _comfy_listener_sockets()):
        raise RuntimeError("無法確認此 ComfyUI 服務由 Studio 啟動")
    queue = call(COMFY_URL + "/queue", timeout=5)
    if not isinstance(queue, dict) or any(queue.get(key) != [] for key in ("queue_running", "queue_pending")):
        raise RuntimeError("ComfyUI 仍有工作或無法確認佇列已清空，請在工作完成後重試")

    check_cancel()
    try:
        pidfd = os.pidfd_open(record["pid"])
    except (AttributeError, OSError) as exc:
        raise RuntimeError("無法取得 ComfyUI 程序身分以安全重啟") from exc
    try:
        if not _owns_comfy_process(record) or not _process_has_listener(proc, _comfy_listener_sockets()):
            raise RuntimeError("ComfyUI 程序身分已變更，請重試")
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        for _ in range(15):
            if select.select([pidfd], [], [], 1)[0]:
                break
            check_cancel()
        else:
            raise RuntimeError("未確認 ComfyUI 程序已停止")
    finally:
        os.close(pidfd)
    if _comfy_system() is not None:
        raise RuntimeError("ComfyUI 端點已有其他服務，請確認後重試")


def _stop_comfy_for_install():
    record = _read_record(COMFY_PROCESS) or {}
    owned = _owns_comfy_process(record)
    if _comfy_system() is None:
        if owned:
            raise RuntimeError("Studio 的 ComfyUI 程序尚未可用，請稍後重試安裝")
        return
    if not owned:
        raise RuntimeError("更新環境前需先停止既有 ComfyUI，Studio 無法確認程序由自己啟動")
    _stop_owned_comfy(record)


def _generation_pixels(graph: dict, references: list[str]) -> int | None:
    latent_id, _ = graph["54"]["inputs"]["latent_image"]
    latent = graph[latent_id]
    pixels = 0
    if latent["class_type"] == "EmptyLatentImage":
        canvas = latent["inputs"]
        pixels = canvas["width"] * canvas["height"]
    if not references:
        return pixels

    # Edit can encode larger references than its output canvas.
    resolution = graph["40"]["inputs"]["resolution"]
    if resolution:
        return max(pixels, resolution**2)

    from PIL import Image

    for ref in references:
        check_cancel()
        try:
            with Image.open(confined_path(JOB / "inputs", ref)) as image:
                width, height = image.size
        except (OSError, ValueError, Image.DecompressionBombError):
            return None
        # Round up for the estimate; the graph aligns the original dimensions to 32.
        aligned_width = ((width + 31) // 32) * 32
        aligned_height = ((height + 31) // 32) * 32
        pixels = max(pixels, aligned_width * aligned_height)
    return pixels


def _batch_fits_gpu(plugin, manifest: dict, cli_quant: str | None, prepared_models: dict) -> bool:
    vram_gib = (STATE.get("gpu_info") or {}).get("vram_gib") or 0
    if not vram_gib:
        return False
    for job in manifest["jobs"]:
        check_cancel()
        models = plugin.required_models_for_batch(dict(manifest, jobs=[job]), cli_quant=cli_quant)
        sizes = [prepared_models.get(model, {}).get("bytes") for model in models]
        if any(type(size) is not int or size <= 0 for size in sizes):
            return False

        graph = plugin.build_qwen21_graph(job, manifest=manifest, cli_quant=cli_quant)
        pixels = _generation_pixels(graph, job.get("inputs", []))
        if pixels is None:
            return False

        # L4 peaks: INT8 2K T2I used ~4 GiB more than 1K T2I.
        # At 1K, the largest measured surplus over weight sizes was
        # ~1.7 GiB for INT8 Edit and ~5.7 GiB for Q4_K_M Edit.
        # Reserve 5.5 GiB for native 1K or 7 GiB for GGUF 1K.
        working_gib = 4 + 1.5 * max(1, pixels / 1024**2)
        if any(folder == "unet" for _, _, folder in models):
            working_gib += 1.5
        if sum(sizes) / 1024**3 + working_gib > vram_gib:
            return False
    return True


def start_comfy(*, gpu_only: bool, environment: str) -> dict:
    check_cancel()
    port = _comfy_loopback_port()
    if port is None:
        raise RuntimeError("ComfyUI 端點必須使用本機 loopback，才能核對執行環境")
    vram_flags = ["--gpu-only"] if gpu_only else []
    argv = [str(COMFY.resolve() / "main.py"), "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch", *vram_flags]
    running = _comfy_system()
    record = _read_record(COMFY_PROCESS) or {}
    owned = _owns_comfy_process(record)
    if running is not None:
        if owned:
            proc = Path("/proc") / str(record["pid"])
            if _process_has_listener(proc, _comfy_listener_sockets()):
                if running["args"] == argv and record.get("environment") == environment:
                    log("comfy: verified running environment and launch arguments")
                    return dict(running, verified=True)
            _stop_owned_comfy(record)
        elif running["args"] in (argv, ["main.py", *argv[1:]]) and _legacy_comfy_checkout():
            log("comfy: verified existing checkout path and launch arguments")
            return dict(running, verified=False)
        else:
            raise RuntimeError("既有 ComfyUI 路徑或啟動參數不符，Studio 無法確認程序由自己啟動")
    elif owned:
        raise RuntimeError("Studio 的 ComfyUI 程序尚未可用，請稍後重試")

    log(f"comfy: starting ComfyUI server ({' '.join(vram_flags) or 'normal-vram'})", "comfy")
    token = uuid.uuid4().hex
    with COMFY_LOG.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, *argv], cwd=str(COMFY), stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, COMFY_INSTANCE: token},
        )
    try:
        _write_record(COMFY_PROCESS, {"pid": process.pid, "token": token, "environment": environment})
    except OSError:
        process.terminate()
        raise
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        check_cancel()
        if process.poll() is not None:
            break
        try:
            running = _comfy_system()
        except (OSError, ComfyHTTPError):
            running = None
        if running is not None:
            proc = Path("/proc") / str(process.pid)
            if running["args"] != argv or not _process_has_listener(proc, _comfy_listener_sockets()):
                raise RuntimeError("ComfyUI 啟動後的程序或參數與預期不符")
            log("comfy: ready")
            return dict(running, verified=True)
        time.sleep(2)
    tail = COMFY_LOG.read_text(errors="ignore")[-2000:]
    raise RuntimeError("ComfyUI did not become ready:\n" + tail)


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


def _remaining_prompt_time(deadline: float, what: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(f"{what}: 已達單項時間上限")
    return remaining


def queue_prompt(graph: dict, *, pin_models: bool = False, deadline: float | None = None) -> str:
    fitted = fit_names(graph)
    if pin_models and fitted:
        raise RuntimeError("ComfyUI 模型路徑偏離已驗證的檔名：" + ", ".join(new for _, _, _, new in fitted))
    check_cancel()
    request_options = ({"timeout": min(60, _remaining_prompt_time(deadline, "ComfyUI 排程"))}
                       if deadline is not None else {})
    pid = str(uuid.uuid4())
    IN_FLIGHT[pid] = False
    STATE["prompt_ids"].append(pid)
    save()
    try:
        resp = call(COMFY_URL + "/prompt", {"prompt": graph, "client_id": "cstudio", "prompt_id": pid},
                    **request_options)
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


def wait_history(pid: str, timeout: float, what: str) -> dict:
    t0 = last = time.monotonic()
    deadline = t0 + timeout
    fails = 0
    while (remaining := deadline - time.monotonic()) > 0:
        cancelling = cancellation_requested()
        if cancelling:
            cancel_prompts()
        try:
            h = call(COMFY_URL + f"/history/{pid}", timeout=min(60, remaining)).get(pid)
            fails = 0
        except Exception as e:
            if cancelling:
                raise BatchCancelled() from e
            fails += 1
            if fails > 5:
                raise RuntimeError(f"{what}: 輪詢 ComfyUI 連續失敗 {fails} 次：{e}")
            time.sleep(max(0, min(2, deadline - time.monotonic())))
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
        if time.monotonic() - last > 30:
            last = time.monotonic()
            log(f"{what}: generating ({time.monotonic() - t0:.0f}s)")
        time.sleep(max(0, min(1.5, deadline - time.monotonic())))
    raise RuntimeError(f"{what}: 超過 {timeout:.0f} 秒未完成")


def stage_input_image(rel_path: str, job_id: str, idx: int, *, expected_sha: str | None = None) -> str:
    """Copy a verified reference from this run into ComfyUI's input directory."""
    input_root = confined_path(JOB, "inputs", directory=True)
    src = confined_path(input_root, rel_path)
    ext = src.suffix.lower() or ".png"
    prefix = f"cstudio_{MANAGED_RUN_ID}_{job_id}" if MANAGED_RUN_ID else f"cstudio_{job_id}"
    safe_name = f"{prefix}_in{idx}{ext}"
    dest = confined_path(COMFY, f"input/{safe_name}", missing_ok=True)
    dest.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".cstudio-reference-") as directory:
        staged = Path(directory) / safe_name
        shutil.copy2(src, staged)
        if expected_sha is not None and file_sha256(staged) != expected_sha:
            raise RuntimeError(f"ComfyUI 參考圖與此批次的 SHA-256 不符：{rel_path}")
        staged.replace(dest)
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
    manifest_bytes = manifest_path.read_bytes()
    data = parse_manifest(manifest_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("遠端 manifest 頂層必須是 JSON object")
    run_id = data.pop("_run_id", None)
    if MANAGED_RUN_ID and run_id != MANAGED_RUN_ID:
        raise RuntimeError("遠端 manifest 與執行中的批次不符")
    if run_id is not None and (not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)):
        raise ValueError("遠端 manifest 的批次 ID 不合法")
    STATE["run_id"] = run_id
    save()
    check_cancel()
    cli_quant = data.pop("_cli_quant", None)
    pin_versions = data.pop("_pin_versions", True)
    wants_model_pins = data.pop("_pin_models", True)
    fingerprint_version = data.pop("_fingerprint_version", 1)
    if type(pin_versions) is not bool or type(wants_model_pins) is not bool:
        raise ValueError("遠端 manifest 的釘選設定必須是布林值")
    if type(fingerprint_version) is not int or fingerprint_version not in (1, 2):
        raise ValueError("遠端 manifest 的指紋版本不受支援")
    manifest, problems = normalize_manifest(data, JOB / "inputs", cli_quant)
    if problems:
        raise ValueError("遠端 manifest 驗證未通過：\n- " + "\n- ".join(problems))
    cli_quant = normalize_quant(cli_quant)
    jobs = manifest["jobs"]
    report_execution = fingerprint_version == 2
    input_shas = {}
    if MANAGED_RUN_ID:
        request = _read_record(confined_path(JOB, "request.json")) or {}
        if (
            request.get("bundle_hash") != STATE.get("bundle_hash")
            or request.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
            or not isinstance(request.get("input_shas"), dict)
        ):
            raise RuntimeError("遠端批次與啟動時驗證的 manifest 或程式版本不符")
        input_shas = request["input_shas"]
        for image in {rel for job in jobs for rel in job.get("inputs") or []}:
            digest = input_shas.get(image)
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise RuntimeError("遠端批次缺少已驗證的參考圖摘要")
    pins = load_pins({"_pin_versions": pin_versions})
    pin_models = pins is not None and wants_model_pins
    plugin = get_model_plugin(manifest.get("model"))

    # 1. Install ComfyUI & ComfyUI-GGUF
    environment = install(pins)
    check_cancel()

    # 2. Download required models for this batch
    req_models = plugin.required_models_for_batch(manifest, cli_quant=cli_quant)
    prepared_models = download_models(req_models, pins=pins if pin_models else None)
    check_cancel()

    # 3. Start ComfyUI
    has_pe = any(job.get("refine_prompt", manifest.get("refine_prompt", False)) for job in jobs)
    gpu_only = not has_pe and _batch_fits_gpu(plugin, manifest, cli_quant, prepared_models)
    comfy = start_comfy(gpu_only=gpu_only, environment=environment)

    confined_path(JOB, OUT.name, directory=True, missing_ok=True).mkdir(exist_ok=True)
    confined_path(JOB, WORK.name, directory=True, missing_ok=True).mkdir(exist_ok=True)

    log(f"batch: starting {len(jobs)} job(s)", "running")

    for idx, job in enumerate(jobs, 1):
        check_cancel()
        jid = job["id"]
        t_job = time.time()
        transparent_pe = plugin.uses_separate_transparent_t2i_pe(job, manifest)
        separate_pe = transparent_pe or plugin.uses_separate_edit_pe(job, manifest)
        refine_on = bool(job.get("refine_prompt", manifest.get("refine_prompt", False)))
        timeout_s = int(job.get("timeout", 1200 if refine_on else 600))
        deadline = time.monotonic() + timeout_s
        STATE["jobs"][jid] = {"status": "running", "started": t_job}
        save()
        log(f"job [{idx}/{len(jobs)}] {jid}: preparing ({job.get('mode', 't2i')})")

        try:
            staged_inputs: list[str] = []
            for i_in, rel_in in enumerate(job.get("inputs") or [], 1):
                staged_inputs.append(stage_input_image(rel_in, jid, i_in, expected_sha=input_shas.get(rel_in)))

            graph = plugin.build_qwen21_graph(
                job,
                manifest=manifest,
                cli_quant=cli_quant,
                uploaded_inputs=staged_inputs,
            )
            save_node = next(nid for nid, node in graph.items() if node["class_type"] == "SaveImage")
            prefix = f"cstudio_{run_id}_{jid}" if MANAGED_RUN_ID else graph[save_node]["inputs"]["filename_prefix"]
            refined_text = None
            pe_receipt = None
            if separate_pe:
                pe_graph = plugin.build_pe_graph(graph, prefix)
                pe_pid = queue_prompt(pe_graph, pin_models=pin_models, deadline=deadline)
                pe_history = wait_history(pe_pid, timeout=_remaining_prompt_time(deadline, jid), what=f"{jid} PE")
                raw = extract_refined_prompt(pe_history)
                if transparent_pe:
                    refined_text = plugin.normalize_transparent_t2i_pe_answer(raw, job["prompt"])
                else:
                    refined_text = plugin.normalize_edit_pe_answer(raw, job["prompt"], len(staged_inputs))
                pe_receipt = {"prompt_id": pe_pid, "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                              "answer_sha256": hashlib.sha256(refined_text.encode()).hexdigest()}
                graph = plugin.build_qwen21_graph(
                    {**job, "prompt": refined_text, "refine_prompt": False}, manifest=manifest,
                    cli_quant=cli_quant, uploaded_inputs=staged_inputs,
                )
            graph[save_node]["inputs"]["filename_prefix"] = prefix
            if separate_pe:
                pid = queue_prompt(graph, pin_models=pin_models, deadline=deadline)
                h = wait_history(pid, timeout=_remaining_prompt_time(deadline, jid), what=f"{jid} image")
            else:
                pid = queue_prompt(graph, pin_models=pin_models)
                h = wait_history(pid, timeout=timeout_s, what=jid)
            if refine_on and not separate_pe:
                refined_text = extract_refined_prompt(h)
                if not refined_text:
                    raise RuntimeError(
                        f"{jid}: refine_prompt=True 但 Prompt Enhancer 輸出為空（可能 thinking 耗盡 token 上限，請調高 pe_max_length 或設 thinking=false）"
                    )
            if refined_text is not None:
                log(f"job {jid}: prompt enhanced ({len(refined_text)} chars)")

            out_file = confined_path(JOB, f"{OUT.name}/{jid}.png", missing_ok=True)
            collect_output_image(h, out_file, save_node=save_node, prefix=graph[save_node]["inputs"]["filename_prefix"])
            sha256 = file_sha256(out_file)
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
            if report_execution:
                job_models = plugin.required_models_for_batch(dict(manifest, jobs=[job]), cli_quant=cli_quant)
                job_rec["execution"] = {
                    "pin_versions": pins is not None, "pin_models": pin_models,
                    "environment": STATE["environment"], "gpu": STATE["gpu_info"], "comfy": comfy,
                    "models": [prepared_models[model] for model in job_models],
                    "fitted_models": (FITTED.get(pe_pid, []) if pe_receipt is not None else []) + FITTED.get(pid, []),
                }
                if pe_receipt is not None:
                    job_rec["execution"]["pe"] = {**pe_receipt, "image_prompt_id": pid}
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
            if separate_pe and IN_FLIGHT:
                cancel_prompts()
                log(f"job {jid}: 已核對逾時或未確認的 prompt 結束", "running")

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
