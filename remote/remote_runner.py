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
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import get_model_plugin  # noqa: E402

JOB = Path(os.environ.get("CSTUDIO_DIR", "/content/cstudio"))
OUT = JOB / "out"
WORK = JOB / "work"
STATUS = JOB / "status.json"
COMFY = Path(os.environ.get("CSTUDIO_COMFY", "/content/ComfyUI"))
MARK = Path("/content/.cstudio_setup_v1")
COMFY_URL = os.environ.get("CSTUDIO_COMFY_URL", "http://127.0.0.1:8188")

STATE: dict = {
    "stage": "starting",
    "run_id": None,
    "log": [],
    "jobs": {},
    "prompt_ids": [],
    "done": False,
    "error": None,
    "started": time.time(),
}

MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
OBJECT_INFO: dict = {}
FITTED: dict = {}


# ---------------------------------------------------------------- status & helpers
def save():
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATUS)


def log(msg: str, stage: str | None = None):
    if stage:
        STATE["stage"] = stage
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE["log"] = (STATE["log"] + [line])[-60:]
    save()


def sh(args, **kw):
    subprocess.run(args, check=True, **kw)


def _git_head(dest: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def clone(url: str, dest: Path, commit: str | None = None):
    dest = Path(dest)
    if commit is None:
        if not dest.exists():
            sh(["git", "clone", "--depth", "1", url, str(dest)])
        else:
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        return
    if str(commit).startswith("-"):
        raise RuntimeError(f"bad pinned commit {commit!r}")
    if not dest.exists():
        sh(["git", "init", "-q", str(dest)])
        sh(["git", "-C", str(dest), "remote", "add", "origin", url])
    else:
        top = subprocess.run(["git", "-C", str(dest), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != dest.resolve():
            raise RuntimeError(f"{dest} exists but is not its own git repo")
    if subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit], check=False).returncode != 0:
        sh(["git", "-C", str(dest), "fetch", "-q", "origin"])
    sh(["git", "-C", str(dest), "checkout", "-q", "--detach", commit])
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
    sh([*pip_cmd, "-r", str(COMFY / "requirements.txt")])
    for n in nodes:
        req = COMFY / "custom_nodes" / n["name"] / "requirements.txt"
        if req.is_file():
            sh([*pip_cmd, "-r", str(req)])

    extra_pip = [p.get("spec") or p["name"] for p in ((pins or {}).get("pip") or [])] or [
        "huggingface_hub>=0.26.0",
        "hf_xet",
        "gguf>=0.13.0",
    ]
    sh([*pip_cmd, *extra_pip])

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


def download_models(files: list[tuple[str, str, str]], pins: dict | None = None):
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import hf_hub_download

    pin_idx = _model_pin_index(pins)
    # Check free disk space before downloading missing files
    needed_bytes = 0
    for repo, name, folder in files:
        dest = COMFY / "models" / folder / Path(name).name
        if not dest.exists() or dest.stat().st_size <= 1024:
            p = pin_idx.get((repo, name))
            if p and p.get("bytes"):
                needed_bytes += int(p["bytes"])

    free_bytes = shutil.disk_usage("/content").free
    if needed_bytes > 0 and free_bytes < needed_bytes + 3 * (1024**3):
        raise RuntimeError(
            f"Colab 磁碟空間不足：需要約 {needed_bytes / 1e9:.1f} GB，剩餘 {free_bytes / 1e9:.1f} GB"
        )

    for i, (repo, name, folder) in enumerate(files, 1):
        dest = COMFY / "models" / folder / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        p = pin_idx.get((repo, name))
        exp_bytes = int(p["bytes"]) if p and p.get("bytes") else None
        rev = p.get("revision") if p else None

        if dest.exists() and dest.stat().st_size > 1024:
            if exp_bytes is None or dest.stat().st_size == exp_bytes:
                log(f"models [{i}/{len(files)}]: cached {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
                continue
            log(f"models [{i}/{len(files)}]: size mismatch on {dest.name}, re-downloading")
            dest.unlink()

        log(f"models [{i}/{len(files)}]: downloading {dest.name} from {repo}" + (f" @ {rev[:8]}" if rev else ""), "models")
        kw = {"revision": rev} if rev else {}
        got = Path(hf_hub_download(repo_id=repo, filename=name, local_dir="/content/hf", **kw))
        if exp_bytes is not None and got.stat().st_size != exp_bytes:
            raise RuntimeError(f"下載檔案大小不符 {dest.name}: 預期 {exp_bytes} bytes, 實際 {got.stat().st_size} bytes")
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(got, dest)
        except OSError:
            shutil.move(str(got), str(dest))
        log(f"models [{i}/{len(files)}]: ready {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")


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


def start_comfy(has_pe: bool = False):
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


def queue_prompt(graph: dict) -> str:
    fitted = fit_names(graph)
    resp = call(COMFY_URL + "/prompt", {"prompt": graph, "client_id": "cstudio"})
    if not isinstance(resp, dict) or not resp.get("prompt_id"):
        raise RuntimeError(f"ComfyUI 回應沒有 prompt_id：{json.dumps(resp, ensure_ascii=False)[:300]}")
    if resp.get("node_errors"):
        raise RuntimeError(f"ComfyUI node_errors：{json.dumps(resp['node_errors'], ensure_ascii=False)[:400]}")
    pid = resp["prompt_id"]
    if fitted:
        FITTED[pid] = fitted
    STATE.setdefault("prompt_ids", []).append(pid)
    save()
    return pid


def wait_history(pid: str, timeout: int, what: str) -> dict:
    t0, last = time.time(), time.time()
    fails = 0
    while time.time() - t0 < timeout:
        try:
            h = call(COMFY_URL + f"/history/{pid}").get(pid)
            fails = 0
        except Exception as e:
            fails += 1
            if fails > 5:
                raise RuntimeError(f"{what}: 輪詢 ComfyUI 連續失敗 {fails} 次：{e}")
            time.sleep(2)
            continue

        if h and h.get("status", {}).get("completed") is not None:
            if h["status"].get("status_str") == "error":
                msgs = [m[1] for m in h["status"].get("messages", []) if m[0] == "execution_error"]
                m = msgs[0] if msgs else {}
                raise RuntimeError(
                    f"ComfyUI 執行錯誤: {m.get('node_type', '')} {m.get('exception_message', 'error')[:500]}".strip()
                )
            return h
        if time.time() - last > 30:
            last = time.time()
            log(f"{what}: generating ({time.time() - t0:.0f}s)")
        time.sleep(1.5)
    raise RuntimeError(f"{what}: 超過 {timeout} 秒未完成")


def stage_input_image(rel_path: str, job_id: str, idx: int) -> str:
    """Copy an uploaded input image from /content/cstudio/inputs/<rel_path> into ComfyUI/input/."""
    src = JOB / "inputs" / rel_path
    if not src.is_file():
        raise FileNotFoundError(f"找不到輸入圖片：{src}")
    ext = src.suffix.lower() or ".png"
    safe_name = f"cstudio_{job_id}_in{idx}{ext}"
    dest = COMFY / "input" / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def collect_output_image(history_entry: dict, dest_path: Path) -> Path:
    """Locate the generated image in ComfyUI/output (or via /view) and copy it to dest_path."""
    outputs = history_entry.get("outputs") or {}
    for nid, out in outputs.items():
        if not isinstance(out, dict):
            continue
        imgs = out.get("images") or []
        if not imgs:
            continue
        info = imgs[0]
        fn = info.get("filename")
        sub = info.get("subfolder", "")
        ftype = info.get("type", "output")
        if not fn:
            continue
        local_candidate = COMFY / ftype / sub / fn if sub else COMFY / ftype / fn
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if local_candidate.is_file():
            shutil.copy2(local_candidate, dest_path)
            return dest_path
        # Fallback to /view HTTP download
        qs = urllib.parse.urlencode({"filename": fn, "subfolder": sub, "type": ftype})
        with urllib.request.urlopen(f"{COMFY_URL}/view?{qs}", timeout=60) as r:
            dest_path.write_bytes(r.read())
        return dest_path
    raise RuntimeError("ComfyUI 完成但未在 history outputs 找到輸出圖片")


# ---------------------------------------------------------------- 4 main batch execution
def run_batch():
    manifest_path = JOB / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"找不到 {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    STATE["run_id"] = manifest.get("_run_id")
    save()

    pins = load_pins(manifest)
    plugin = get_model_plugin(manifest.get("model"))
    cli_quant = manifest.get("_cli_quant")

    # 1. Install ComfyUI & ComfyUI-GGUF
    install(pins)

    # 2. Download required models for this batch
    req_models = plugin.required_models_for_batch(manifest, cli_quant=cli_quant)
    download_models(req_models, pins=pins if manifest.get("_pin_models", True) else None)

    # 3. Start ComfyUI
    has_pe = any(bool(j.get("refine_prompt", manifest.get("refine_prompt", False))) for j in (manifest.get("jobs") or []))
    start_comfy(has_pe=has_pe)

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    jobs = manifest.get("jobs") or []
    log(f"batch: starting {len(jobs)} job(s)", "running")

    for idx, job in enumerate(jobs, 1):
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
            pid = queue_prompt(graph)
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

            out_file = OUT / f"{jid}.png"
            collect_output_image(h, out_file)
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
    STATE["done"] = True
    STATE["stage"] = "done" if not bad else "done_with_errors"
    if bad:
        STATE["error"] = f"{len(bad)}/{len(jobs)} jobs failed: {', '.join(bad)}"
    save()
    log(f"batch finished: {len(jobs) - len(bad)}/{len(jobs)} succeeded")


if __name__ == "__main__":
    try:
        run_batch()
    except Exception as exc:
        STATE["done"] = True
        STATE["stage"] = "error"
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log(f"FATAL: {STATE['error']}")
        traceback.print_exc()
        save()
        sys.exit(1)
