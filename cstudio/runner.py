# -*- coding: utf-8 -*-
"""Batch validation, fingerprinting, planning, and Colab execution/sync engine."""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from . import colab_client as C

ROOT = Path(__file__).resolve().parent.parent
REMOTE = ROOT / "remote"
sys.path.insert(0, str(REMOTE))
from models import DEFAULT_MODEL, SUPPORTED_MODELS, get_model_plugin  # noqa: E402


def check_ref(base: Path, f: str, jid: str = "") -> str | None:
    s = str(f).replace("\\", "/")
    who = f"{jid}：" if jid else ""
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        return f"{who}檔案路徑含有控制字元或換行：{s!r}"
    if s.startswith(("/", "~")) or re.match(r"[A-Za-z]:", s):
        return f"{who}不允許絕對路徑或 ~ 開頭的路徑：{s}（請用相對於 manifest 的路徑）"
    base = Path(base)
    parent = base.parent.as_posix().rstrip("/")
    norm = posixpath.normpath(posixpath.join(base.as_posix(), s))
    if norm != parent and not norm.startswith(parent + "/"):
        return f"{who}路徑跑出 manifest 上一層資料夾之外：{s}"
    if not posixpath.normpath("/content/cstudio/inputs/" + s).startswith("/content/cstudio/"):
        return f"{who}路徑上傳後會跑出 /content/cstudio/ 之外：{s}"
    try:
        real = (base / s).resolve()
        real_parent = base.parent.resolve()
        if real != real_parent and real_parent not in real.parents:
            return f"{who}不允許指向專案目錄外的符號連結：{s} -> {real}"
    except OSError:
        pass
    return None


def validate_manifest(manifest: dict[str, Any], base_dir: Path, cli_quant: str | None = None) -> list[str]:
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest 頂層必須是 JSON object"]
    model_name = str(manifest.get("model") or DEFAULT_MODEL).strip().lower()
    if model_name not in SUPPORTED_MODELS:
        problems.append(f"不支援的模型 {model_name!r}（可用：{', '.join(SUPPORTED_MODELS)}）")
        return problems

    plugin = get_model_plugin(model_name)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        problems.append("manifest 的 jobs 不能為空")
        return problems

    seen: set[str] = set()
    for j in jobs:
        if not isinstance(j, dict):
            problems.append("jobs 內每個項目都必須是 JSON object")
            continue
        raw_id = j.get("id")
        if not isinstance(raw_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", raw_id):
            problems.append(f"id 必須是字串且只能用英數、- 和 _（1～60 字元）：{raw_id!r}")
            continue
        jid = raw_id
        if jid in seen:
            problems.append(f"id 重複：{jid}")
        seen.add(jid)

        mode = str(j.get("mode", "t2i")).strip().lower()
        if mode not in ("t2i", "edit"):
            problems.append(f"{jid}：mode 必須是 't2i' 或 'edit'（收到 {mode!r}）")

        if not str(j.get("prompt", "")).strip():
            problems.append(f"{jid}：缺少 prompt")

        inputs = j.get("inputs") or []
        if not isinstance(inputs, list):
            problems.append(f"{jid}：inputs 必須是陣列")
            inputs = []
        if mode == "edit" and not (1 <= len(inputs) <= 10):
            problems.append(f"{jid}：edit 模式需要 1～10 張參考圖片（收到 {len(inputs)} 張）")
        if mode == "t2i" and inputs:
            problems.append(f"{jid}：t2i 模式不可提供 inputs（若要改圖請設 mode='edit'）")

        for f in inputs:
            bad = check_ref(base_dir, str(f), jid)
            if bad:
                problems.append(bad)
            elif not (base_dir / str(f)).is_file():
                problems.append(f"{jid}：找不到輸入圖片 {f}（相對於 {base_dir}）")

        # Test graph building offline to catch invalid quant, aspect_ratio, resolution, or PE variant early
        try:
            plugin.build_qwen21_graph(j, manifest=manifest, cli_quant=cli_quant)
        except Exception as e:
            problems.append(f"{jid}：{e}")

    return problems


def load_manifest(path: str | Path, cli_quant: str | None = None) -> tuple[dict[str, Any], Path]:
    p = Path(path).resolve()
    if not p.is_file():
        sys.exit(f"找不到 manifest 檔案：{p}")
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"manifest JSON 格式錯誤（{p}）：{e}")
    base = p.parent
    problems = validate_manifest(m, base, cli_quant=cli_quant)
    if problems:
        sys.exit("manifest 驗證未通過，沒有開機：\n- " + "\n- ".join(problems))
    return m, base


def load_versions() -> dict[str, Any]:
    vf = REMOTE / "versions.json"
    return json.loads(vf.read_text(encoding="utf-8")) if vf.is_file() else {}


def compute_job_fingerprint(
    job: dict[str, Any],
    manifest: dict[str, Any],
    base_dir: Path,
    cli_quant: str | None = None,
    versions: dict[str, Any] | None = None,
) -> str:
    plugin = get_model_plugin(manifest.get("model"))
    graph = plugin.build_qwen21_graph(job, manifest=manifest, cli_quant=cli_quant)

    input_hashes = []
    for rel in job.get("inputs") or []:
        fp = base_dir / str(rel)
        h = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else "missing"
        input_hashes.append({"path": str(rel).replace("\\", "/"), "sha256": h})

    v_commits = {}
    if isinstance(versions, dict):
        v_commits["comfyui"] = (versions.get("comfyui") or {}).get("commit")
        v_commits["nodes"] = {n.get("name"): n.get("commit") for n in (versions.get("custom_nodes") or [])}
        req_for_job = plugin.required_models_for_batch(dict(manifest, jobs=[job]), cli_quant=cli_quant)
        pin_idx = {(e["repo"], e["file"]): e.get("sha256") or e.get("revision") for e in (versions.get("models") or []) if "repo" in e and "file" in e}
        v_commits["models"] = {f"{r}/{fn}": pin_idx.get((r, fn)) for r, fn, _ in req_for_job}

    payload = {
        "graph_version": getattr(plugin, "GRAPH_VERSION", "v1"),
        "graph": graph,
        "inputs": input_hashes,
        "versions": v_commits,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def classify_jobs(
    manifest: dict[str, Any],
    base_dir: Path,
    cli_quant: str | None = None,
    use_fingerprint: bool = True,
) -> tuple[dict[str, str], list[dict[str, Any]], list[str], list[tuple[str, str]]]:
    """Return (fps, todo_jobs, done_ids, stale_list[(jid, reason)])."""
    out_dir = base_dir / "out"
    res_file = out_dir / "results.json"
    old_results = {}
    if res_file.is_file():
        try:
            old_results = (json.loads(res_file.read_text(encoding="utf-8")).get("jobs")) or {}
        except Exception:
            old_results = {}

    versions = load_versions()
    fps: dict[str, str] = {}
    todo: list[dict[str, Any]] = []
    done_ids: list[str] = []
    stale: list[tuple[str, str]] = []

    for j in manifest.get("jobs") or []:
        jid = j["id"]
        fp = compute_job_fingerprint(j, manifest, base_dir, cli_quant=cli_quant, versions=versions)
        fps[jid] = fp
        out_img = out_dir / f"{jid}.png"
        if not out_img.is_file() or out_img.stat().st_size == 0:
            todo.append(j)
            continue
        if not use_fingerprint:
            done_ids.append(jid)
            continue
        prev = old_results.get(jid) or {}
        prev_fp = prev.get("fingerprint")
        if prev_fp == fp:
            done_ids.append(jid)
        elif prev_fp is None:
            stale.append((jid, "舊輸出沒有指紋紀錄"))
            todo.append(j)
        else:
            stale.append((jid, "prompt、量化等級或參考圖片已變更"))
            todo.append(j)

    return fps, todo, done_ids, stale


def plan_batch(manifest_path: str | Path, gpu: str = "L4", cli_quant: str | None = None) -> dict[str, Any]:
    m, base = load_manifest(manifest_path, cli_quant=cli_quant)
    plugin = get_model_plugin(m.get("model"))
    fps, todo, done_ids, stale = classify_jobs(m, base, cli_quant=cli_quant)
    req_models = plugin.required_models_for_batch(m, cli_quant=cli_quant)
    versions = load_versions()
    pin_idx = {(e["repo"], e["file"]): e for e in (versions.get("models") or []) if "repo" in e and "file" in e}

    total_bytes = 0
    model_list = []
    for repo, fn, folder in req_models:
        p = pin_idx.get((repo, fn)) or {}
        b = int(p.get("bytes") or 0)
        total_bytes += b
        model_list.append({"repo": repo, "file": fn, "dest": folder, "gb": round(b / 1e9, 2)})

    print(f"📋 批次規劃：{Path(manifest_path).resolve()}", flush=True)
    print(f"   模型：{m.get('model', DEFAULT_MODEL)} | 目標 GPU：{gpu}", flush=True)
    print(f"   任務總數：{len(m['jobs'])}（已完成略過：{len(done_ids)}，待執行：{len(todo)}，過期：{len(stale)}）", flush=True)
    print(f"   需準備模型檔（共約 {total_bytes / 1e9:.2f} GB）：", flush=True)
    for ml in model_list:
        print(f"     - [{ml['dest']}] {ml['file']} ({ml['gb']:.2f} GB)", flush=True)
    print("   各任務明細：", flush=True)
    for j in m["jobs"]:
        jid = j["id"]
        w, h = plugin.resolve_dimensions(j, m)
        q = (cli_quant or j.get("quant") or m.get("quant") or plugin.DEFAULT_QUANT).upper()
        ref = "✨PE擴寫" if j.get("refine_prompt", m.get("refine_prompt", False)) else "直出"
        rgba = " [RGBA透明]" if j.get("transparent") else ""
        st = "✅ 已完成" if jid in done_ids else ("⚠️ 過期" if any(s[0] == jid for s in stale) else "🆕 待生成")
        print(f"     - {jid}: {st} | mode={j.get('mode', 't2i')}{rgba} | quant={q} | {w}x{h} | {ref}", flush=True)

    return {
        "total": len(m["jobs"]),
        "done": done_ids,
        "todo": [j["id"] for j in todo],
        "stale": stale,
        "models": model_list,
        "total_gb": round(total_bytes / 1e9, 2),
    }


def makedirs_code(dirs: list[str]) -> str:
    return (
        "import os, json\n"
        "for d in json.loads(" + repr(json.dumps(dirs)) + "):\n"
        "    os.makedirs('/content/cstudio/inputs/' + d, exist_ok=True)\n"
    )


def save_results(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


REMOTE_CODE_FILES = (
    "__init__.py",
    "remote_runner.py",
    "start.py",
    "status.py",
    "versions.json",
    "models/__init__.py",
    "models/qwen_image_21.py",
    "models/qwen_prompts.py",
)


def compute_remote_bundle_hash() -> str:
    h = hashlib.sha256()
    for rel in REMOTE_CODE_FILES:
        h.update(rel.encode("utf-8"))
        h.update((REMOTE / rel).read_bytes())
    return h.hexdigest()


def bundle_and_launch_code(
    code_files: dict[str, str] | None,
    bundle_hash: str,
    manifest_json: str,
    input_dirs: list[str],
    input_shas: dict[str, str],
) -> str:
    """Single-roundtrip Python code executed on Colab VM:
    1. Creates /content/cstudio directories and input subdirs
    2. Writes remote code files if code_files is provided (when bundle_hash changed)
    3. Writes /content/cstudio/manifest.json
    4. Checks which input files already exist on Colab with matching SHA256
    5. If no input files are missing, starts remote_runner.py immediately!
    """
    payload = {
        "code_files": code_files,
        "bundle_hash": bundle_hash,
        "manifest_json": manifest_json,
        "input_dirs": input_dirs,
        "input_shas": input_shas,
    }
    return (
        "import hashlib, json, os, shutil, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"P = json.loads({json.dumps(payload, ensure_ascii=False)!r})\n"
        "JOB = Path('/content/cstudio')\n"
        "(JOB / 'models').mkdir(parents=True, exist_ok=True)\n"
        "(JOB / 'inputs').mkdir(parents=True, exist_ok=True)\n"
        "for d in P['input_dirs']:\n"
        "    (JOB / 'inputs' / d).mkdir(parents=True, exist_ok=True)\n"
        "if P['code_files']:\n"
        "    for rel, content in P['code_files'].items():\n"
        "        dst = JOB / rel\n"
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "        dst.write_text(content, encoding='utf-8')\n"
        "    (JOB / '.bundle_sha256').write_text(P['bundle_hash'], encoding='utf-8')\n"
        "(JOB / 'manifest.json').write_text(P['manifest_json'], encoding='utf-8')\n"
        "missing = []\n"
        "for rel, exp_sha in P['input_shas'].items():\n"
        "    fp = JOB / 'inputs' / rel\n"
        "    if not fp.is_file() or hashlib.sha256(fp.read_bytes()).hexdigest() != exp_sha:\n"
        "        missing.append(rel)\n"
        "res = {'missing_inputs': missing, 'launched': False}\n"
        "if not missing:\n"
        "    running = subprocess.run(['pgrep', '-f', 'cstudio/remote_runner.py'], capture_output=True, text=True).stdout.strip()\n"
        "    out_dir = JOB / 'out'\n"
        "    status = JOB / 'status.json'\n"
        "    if running:\n"
        "        res['launched'] = True\n"
        "        res['already_running'] = True\n"
        "        res['pid'] = running\n"
        "    else:\n"
        "        if status.exists():\n"
        "            status.rename(JOB / f'status_{int(time.time())}.json')\n"
        "        if out_dir.exists():\n"
        "            shutil.rmtree(out_dir, ignore_errors=True)\n"
        "        out_dir.mkdir(parents=True, exist_ok=True)\n"
        "        p = subprocess.Popen([sys.executable, str(JOB / 'remote_runner.py')], cwd=str(JOB),\n"
        "                             stdout=open(JOB / 'remote_runner.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)\n"
        "        res['launched'] = True\n"
        "        res['pid'] = p.pid\n"
        "print('CSTUDIO_PREP ' + json.dumps(res))\n"
    )


def run_manifest(
    manifest_path: str | Path,
    gpu: str = "L4",
    session: str | None = None,
    cli_quant: str | None = None,
    idle_minutes: float = 15.0,
    max_minutes: float = 120.0,
    stop_after: bool = False,
    force: bool = False,
    remake_stale: bool = False,
    use_fingerprint: bool = True,
    pin_versions: bool = True,
    pin_models: bool = True,
    assign_retries: int = 5,
    assign_wait: int = 120,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if gpu not in C.GPUS:
        sys.exit(f"不支援的 GPU {gpu!r}（可用：{', '.join(C.GPUS)}）")
    if not C.token_ok():
        sys.exit("尚未登入 Google Colab：請先執行 `uv run cstudio login url`")

    m, base = load_manifest(manifest_path, cli_quant=cli_quant)
    out_dir = base / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = out_dir / "results.json"
    prev_results = {}
    if res_file.is_file():
        try:
            prev_results = json.loads(res_file.read_text(encoding="utf-8"))
        except Exception:
            prev_results = {}

    fps, todo, done_ids, stale = classify_jobs(m, base, cli_quant=cli_quant, use_fingerprint=use_fingerprint)
    if force:
        todo = list(m["jobs"])
        done_ids = []
        stale = []
    elif stale and not remake_stale:
        msg = "發現已存在的輸出與目前設定不符（過期）：\n" + "\n".join(f"  - {jid}: {reason}" for jid, reason in stale)
        msg += "\n👉 若要重做過期項目（舊檔會移至 out/_stale/），請加上 `--remake-stale`；若要全部重做請加 `--force`。"
        sys.exit(msg)

    if stale and remake_stale:
        stale_dir = out_dir / "_stale"
        stale_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        for jid, _ in stale:
            old_f = out_dir / f"{jid}.png"
            if old_f.is_file():
                shutil.move(str(old_f), str(stale_dir / f"{jid}_{ts}.png"))

    if not todo:
        print(f"✅ 全部 {len(m['jobs'])} 個任務皆已完成且指紋相符，無需啟動 Colab GPU！（輸出位於 {out_dir}）", flush=True)
        return prev_results

    if done_ids:
        print(f"⏭️ 略過已完成且指紋相符的任務：{', '.join(done_ids)}", flush=True)

    cu_before, _ = C.usage()
    alive, _ = C.sessions()
    created = False

    if not session or session not in alive:
        session = session or f"{C.PREFIX}{time.strftime('%m%d-%H%M%S')}"
        print(f"🚀 在 Google Colab 開啟 {gpu}：{session}", flush=True)
        try:
            C.new_runtime(session, gpu, retries=assign_retries, wait=assign_wait)
            created = True
            C.set_keep(session, None, owner="owned")
        except BaseException:
            print(f"⚠️ 開機過程出錯，嘗試關閉可能已建立的 {session}", flush=True)
            try:
                C.colab("stop", "-s", session, timeout=300, check=False)
                C.drop_keep(session)
            except Exception:
                pass
            raise
        owned = True
    else:
        owned = C.is_owned_session(session)
        st_check = C.poll_remote_status(session)
        if st_check and st_check.get("alive"):
            sys.exit(f"⚠️ Session {session} 上已有正在執行的批次任務，請先等待完成或執行 `uv run cstudio cancel --session {session}`")
        print(f"♻️ 沿用開啟中的 Colab Session：{session}（省去安裝與重複下載模型）", flush=True)
        C.set_keep(session, None, owner="owned" if owned else "external")

    t0 = time.time()
    run_id = uuid.uuid4().hex[:12]
    results: dict[str, Any] = {
        "session": session,
        "gpu": gpu,
        "run_id": run_id,
        "model": m.get("model", DEFAULT_MODEL),
        "jobs": dict((prev_results.get("jobs") or {}) if not force else {}),
    }

    short = dict(timeout=45, tries=6)
    got: set[str] = set()
    finished = False

    try:
        # Prepare remote manifest containing only `todo` jobs + metadata flags
        remote_m = dict(
            m,
            jobs=todo,
            _run_id=run_id,
            _pin_versions=bool(pin_versions),
            _pin_models=bool(pin_models and pin_versions),
        )
        if cli_quant:
            remote_m["_cli_quant"] = cli_quant
        manifest_json = json.dumps(remote_m, ensure_ascii=False, indent=1)

        bundle_hash = compute_remote_bundle_hash()
        k_info = C.keep_info(session) or {}
        need_code = created or k_info.get("bundle_hash") != bundle_hash
        code_files = (
            {rel: (REMOTE / rel).read_text(encoding="utf-8") for rel in REMOTE_CODE_FILES}
            if need_code
            else None
        )

        files = sorted({str(f).replace("\\", "/") for j in todo for f in (j.get("inputs") or [])})
        dirs = sorted({str(Path(f).parent).replace("\\", "/") for f in files} - {"."})
        input_shas = {f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in files}

        prep_out = C.remote_exec(
            session,
            bundle_and_launch_code(code_files, bundle_hash, manifest_json, dirs, input_shas),
            **short,
        )
        if need_code:
            old_k = C.keep_info(session) or {}
            old_k["bundle_hash"] = bundle_hash
            C.keep_file(session).write_text(json.dumps(old_k), encoding="utf-8")

        prep_info = {}
        for line in reversed(prep_out.splitlines()):
            if "CSTUDIO_PREP " in line:
                try:
                    prep_info = json.loads(line.split("CSTUDIO_PREP ", 1)[1])
                except Exception:
                    pass
                break

        missing_inputs = prep_info.get("missing_inputs", files)
        for f in missing_inputs:
            print(f"⬆️ 上傳參考圖：{f}", flush=True)
            C.colab("upload", "-s", session, base / f, f"/content/cstudio/inputs/{f}", timeout=600)

        if prep_info.get("launched"):
            print(f"▶️ 已單次指令同步並啟動遠端引擎 (pid={prep_info.get('pid')})", flush=True)
        else:
            launch_out = C.remote_exec(session, (REMOTE / "start.py").read_text(encoding="utf-8"), **short)
            print(f"▶️ 已啟動遠端生成引擎：{launch_out.strip()[-200:]}", flush=True)

        shown_log: set[str] = set()
        last_ok = time.time()

        while True:
            time.sleep(4)
            if time.time() - t0 > max_minutes * 60:
                raise RuntimeError(f"超過 {max_minutes} 分鐘上限，中止以保護運算單元")

            st = C.poll_remote_status(session)
            if not st:
                if time.time() - last_ok > 15 * 60:
                    raise RuntimeError("連續 15 分鐘無法讀取 Colab 狀態，連線可能已中斷")
                print("…等待 Colab 回應中", flush=True)
                continue

            if st.get("run_id") and st.get("run_id") != run_id:
                continue

            last_ok = time.time()
            for line in st.get("log") or []:
                if line not in shown_log:
                    shown_log.add(line)
                    print(f"  [Colab] {line}", flush=True)

            if progress_cb:
                try:
                    progress_cb(st)
                except Exception:
                    pass

            # Pull newly completed images immediately via atomic temporary file + sha256 verification
            r_jobs = st.get("jobs") or {}
            for jid, jinfo in r_jobs.items():
                if jid in got or not isinstance(jinfo, dict):
                    continue
                if jinfo.get("status") == "done" and jinfo.get("file"):
                    tmp_pull = out_dir / f".{jid}_{run_id}.png.tmp"
                    final_dst = out_dir / f"{jid}.png"
                    max_pull_tries = 3 if st.get("done") else 1
                    for pull_try in range(max_pull_tries):
                        try:
                            C.colab("download", "-s", session, f"/content/cstudio/{jinfo['file']}", tmp_pull, timeout=600)
                            if not tmp_pull.is_file() or tmp_pull.stat().st_size == 0:
                                raise RuntimeError("下載的暫存檔為空")
                            exp_sha = jinfo.get("sha256")
                            got_sha = hashlib.sha256(tmp_pull.read_bytes()).hexdigest()
                            if exp_sha and got_sha != exp_sha:
                                raise RuntimeError(f"SHA256 校驗不符 (expected {exp_sha[:10]}, got {got_sha[:10]})")
                            tmp_pull.replace(final_dst)
                            rec = dict(jinfo, fingerprint=fps.get(jid))
                            results["jobs"][jid] = rec
                            save_results(res_file, results)
                            got.add(jid)
                            print(f"⬇️ 已同步回本地：{final_dst} ({final_dst.stat().st_size / 1024:.1f} KB)", flush=True)
                            if rec.get("refined_prompt"):
                                print(f"   ✨ Prompt Enhancer 擴寫結果：{rec['refined_prompt'][:160]}...", flush=True)
                            break
                        except Exception as e:
                            tmp_pull.unlink(missing_ok=True)
                            if pull_try + 1 < max_pull_tries:
                                print(f"⚠️ 拉回 {jid}.png 失敗，重試中 ({pull_try + 1}/{max_pull_tries})：{e}", flush=True)
                                time.sleep(2)
                            else:
                                print(f"⚠️ 拉回 {jid}.png 失敗：{e}", flush=True)
                elif jinfo.get("status") == "error":
                    results["jobs"][jid] = dict(jinfo)
                    save_results(res_file, results)
                    got.add(jid)
                    print(f"❌ 任務 {jid} 失敗：{jinfo.get('error')}", flush=True)

            if st.get("done"):
                if st.get("error") and st.get("stage") in ("error", "crashed", "cancelled"):
                    results["error"] = st["error"]
                finished = True
                break

    finally:
        results["minutes"] = round((time.time() - t0) / 60, 2)
        try:
            if stop_after or (not finished and created):
                print(f"🛑 關閉 Colab GPU：{session}", flush=True)
                C.colab("stop", "-s", session, timeout=300, check=False)
                names, lout = C.sessions_checked()
                if C.session_gone(session, names, lout):
                    C.drop_keep(session)
                else:
                    C.set_keep(session, time.time() - 1, owner="owned")
                    C.start_watchdog(session)
            elif not finished:
                if owned or created:
                    C.set_keep(session, time.time() + idle_minutes * 60, owner="owned")
                    C.start_watchdog(session)
                else:
                    C.drop_keep(session)
                print(
                    f"⚠️ 批次未完成，{session} 仍在運行。"
                    f"若不再使用請執行：uv run cstudio stop --session {session}",
                    flush=True,
                )
            else:
                C.set_keep(
                    session,
                    time.time() + idle_minutes * 60,
                    owner="owned" if (owned or created) else "external",
                )
                C.start_watchdog(session)
                print(f"⏸️ Colab GPU 保留待命中：{session}（模型已快取在雲端，下一批免重新下載）", flush=True)
                print(f"   👉 若 {idle_minutes:g} 分鐘內無新任務，背景守護進程將自動關機以保護點數。", flush=True)
                print(f"   續跑下一批：uv run cstudio run <manifest.json> --session {session}", flush=True)
                print(f"   延長保留  ：uv run cstudio keep --session {session} --minutes 60", flush=True)
                print(f"   立即關機  ：uv run cstudio stop --session {session}", flush=True)
        finally:
            cu_after, _ = C.usage()
            cu_used = round(cu_before - cu_after, 2) if (cu_before is not None and cu_after is not None) else None
            results.update(cu_before=cu_before, cu_after=cu_after, cu_used=cu_used)
            save_results(res_file, results)
            if cu_used is not None:
                print(f"💳 本批預估消耗 {cu_used} 運算單元（{cu_before} → {cu_after}；Colab 用量可能延遲更新）", flush=True)

    expected_ids = [j["id"] for j in m["jobs"]]
    bad = [
        jid
        for jid in expected_ids
        if (results["jobs"].get(jid) or {}).get("status") != "done" or not (out_dir / f"{jid}.png").is_file()
    ]
    print(f"✅ 批次結束：成功 {len(expected_ids) - len(bad)}/{len(expected_ids)}，耗時 {results['minutes']} 分鐘 → {out_dir}", flush=True)
    if bad or results.get("error"):
        sys.exit(1)
    return results
