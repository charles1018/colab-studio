# -*- coding: utf-8 -*-
"""Batch validation, fingerprinting, planning, and Colab execution/sync engine."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from remote.job_paths import JOB_ID, bundle_digest, confined_path

from . import colab_client as C
from .redact import redact, redact_diagnostics

ROOT = Path(__file__).resolve().parent.parent
REMOTE = ROOT / "remote"
sys.path.insert(0, str(REMOTE))
from models import DEFAULT_MODEL, SUPPORTED_MODELS, get_model_plugin  # noqa: E402


def check_ref(base: Path, f: str, jid: str = "") -> str | None:
    who = f"{jid}：" if jid else ""
    try:
        confined_path(base, f)
    except FileNotFoundError:
        return f"{who}找不到輸入圖片 {f!r}（相對於 {base}）"
    except (OSError, ValueError) as exc:
        return f"{who}無法使用輸入圖片 {f!r}：{exc}"
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
        if not isinstance(raw_id, str) or not JOB_ID.fullmatch(raw_id):
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
            bad = check_ref(base_dir, f, jid)
            if bad:
                problems.append(bad)

        # Test graph building offline to catch invalid quant, aspect_ratio, resolution, or PE variant early
        try:
            plugin.build_qwen21_graph(j, manifest=manifest, cli_quant=cli_quant)
        except Exception as e:
            problems.append(f"{jid}：{e}")

    return problems


def _fail(msg: str, exit_on_error: bool = True):
    if exit_on_error:
        sys.exit(msg)
    raise RuntimeError(msg)


def load_manifest(
    path: str | Path,
    cli_quant: str | None = None,
    exit_on_error: bool = True,
) -> tuple[dict[str, Any], Path]:
    p = Path(path).absolute()
    try:
        base = p.parent.resolve()
        p = confined_path(base, p.name)
    except FileNotFoundError:
        _fail(f"找不到 manifest 檔案：{p}", exit_on_error)
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"無法使用 manifest 檔案 {p}：{exc}", exit_on_error)
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _fail(f"manifest JSON 格式錯誤（{p}）：{e}", exit_on_error)
    problems = validate_manifest(m, base, cli_quant=cli_quant)
    if problems:
        _fail("manifest 驗證未通過，沒有開機：\n- " + "\n- ".join(problems), exit_on_error)
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
        fp = confined_path(base_dir, rel, missing_ok=True)
        h = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else "missing"
        input_hashes.append({"path": rel, "sha256": h})

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
    res_file = confined_path(base_dir, "out/results.json", missing_ok=True)
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
        out_img = confined_path(base_dir, f"out/{jid}.png", missing_ok=True)
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
            stale.append((jid, "生成流程、參考圖片或版本已變更"))
            todo.append(j)

    return fps, todo, done_ids, stale


def plan_batch(manifest_path: str | Path, gpu: str = "L4", cli_quant: str | None = None) -> dict[str, Any]:
    m, base = load_manifest(manifest_path, cli_quant=cli_quant)
    plugin = get_model_plugin(m.get("model"))
    try:
        fps, todo, done_ids, stale = classify_jobs(m, base, cli_quant=cli_quant)
    except (OSError, ValueError) as exc:
        _fail(f"無法使用任務檔案：{exc}")
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


def save_results(path: Path, data: dict[str, Any]):
    confined_path(path.parent, path.name, missing_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as fp:
            tmp = Path(fp.name)
            json.dump(data, fp, ensure_ascii=False, indent=2)
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


REMOTE_CODE_FILES = (
    "__init__.py",
    "cancel.py",
    "job_paths.py",
    "remote_runner.py",
    "versions.json",
    "models/__init__.py",
    "models/qwen_image_21.py",
    "models/qwen_prompts.py",
)


def compute_remote_bundle_hash() -> str:
    return bundle_digest({rel: (REMOTE / rel).read_text(encoding="utf-8") for rel in REMOTE_CODE_FILES})


def bundle_and_launch_code(
    code_files: dict[str, str] | None,
    bundle_hash: str,
    manifest_json: str,
    input_shas: dict[str, str],
) -> str:
    """Prepare a run and start it immediately when there are no references to upload."""
    return C.vm_program(
        "PREP", "prepare_run", bundle_hash, list(REMOTE_CODE_FILES), code_files, manifest_json, input_shas,
        include_start=True,
    )


def _prepare_result(output: str, run_id: str, bundle_hash: str, input_shas: dict[str, str]) -> dict:
    result = None
    for line in reversed(output.splitlines()):
        if line.startswith("CSTUDIO_PREP "):
            try:
                result = json.loads(line.partition(" ")[2])
            except ValueError:
                pass
            break
    if not isinstance(result, dict) or result.get("run_id") != run_id:
        raise RuntimeError("收到無法驗證的遠端準備結果，批次 ID 不符")
    state = result.get("state")
    if state in ("bundle_required", "busy"):
        return result
    if state not in ("prepared", "running", "finished") or result.get("bundle_hash") != bundle_hash:
        raise RuntimeError("收到無法驗證的遠端準備結果，狀態或程式版本不符")
    missing = result.get("missing_inputs")
    if (
        not isinstance(missing, list)
        or any(not isinstance(name, str) or name not in input_shas for name in missing)
        or len(set(missing)) != len(missing) or (state != "prepared" and missing)
    ):
        raise RuntimeError("遠端待上傳清單不符合當批參考圖片")
    return result


class RunCancelled(RuntimeError):
    """The batch was cancelled locally or confirmed cancelled on the VM."""


class RunStopped(RuntimeError):
    """The local batch accepted a shutdown request."""

    outcome: C.ReleaseState | None = None


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
    exit_on_error: bool = True,
    should_abort: Callable[[], str | bool | None] | None = None,
    *,
    account: str | None = None,
    auto_reuse: bool = False,
    reuse_only: bool = False,
) -> dict[str, Any]:
    if gpu not in C.GPUS:
        _fail(f"不支援的 GPU {gpu!r}（可用：{', '.join(C.GPUS)}）", exit_on_error)
    if not math.isfinite(max_minutes * 60) or max_minutes <= 0:
        _fail("批次時間上限必須為有限正數", exit_on_error)
    if not math.isfinite(idle_minutes * 60):
        _fail("GPU 待命時間必須為有限數字", exit_on_error)
    if assign_retries < 0 or assign_wait < 0:
        _fail("GPU 配置的重試次數與等待時間必須為非負數", exit_on_error)
    if session:
        try:
            session = C.sanitize_session_name(session)
        except ValueError as exc:
            _fail(str(exc), exit_on_error)
    if not C.token_ok():
        _fail("尚未登入 Google Colab：請先執行 `uv run cstudio login url`", exit_on_error)

    m, base = load_manifest(manifest_path, cli_quant=cli_quant, exit_on_error=exit_on_error)
    try:
        fps, todo, done_ids, stale = classify_jobs(m, base, cli_quant=cli_quant, use_fingerprint=use_fingerprint)
        out_dir = confined_path(base, "out", directory=True, missing_ok=True)
        out_dir.mkdir(exist_ok=True)
        res_file = confined_path(base, "out/results.json", missing_ok=True)
    except (OSError, ValueError) as exc:
        _fail(f"無法使用任務檔案：{exc}", exit_on_error)
    prev_results = {}
    if res_file.is_file():
        try:
            prev_results = json.loads(res_file.read_text(encoding="utf-8"))
        except Exception:
            prev_results = {}

    if force:
        todo = list(m["jobs"])
        done_ids = []
        stale = []
    elif stale and not remake_stale:
        msg = "發現已存在的輸出與目前設定不符（過期）：\n" + "\n".join(f"  - {jid}: {reason}" for jid, reason in stale)
        msg += "\n👉 若要重做過期項目（舊檔會移至 out/_stale/），請加上 `--remake-stale`；若要全部重做請加 `--force`。"
        _fail(msg, exit_on_error)

    if stale and remake_stale:
        try:
            stale_dir = confined_path(base, "out/_stale", directory=True, missing_ok=True)
            stale_dir.mkdir(exist_ok=True)
            ts = int(time.time())
            for jid, _ in stale:
                old_f = confined_path(base, f"out/{jid}.png", missing_ok=True)
                if old_f.is_file():
                    backup = confined_path(base, f"out/_stale/{jid}_{ts}.png", missing_ok=True)
                    shutil.move(old_f, backup)
        except (OSError, ValueError) as exc:
            _fail(f"無法備份舊成果：{exc}", exit_on_error)

    if not todo:
        print(f"✅ 全部 {len(m['jobs'])} 個任務皆已完成且指紋相符，無需啟動 Colab GPU！（輸出位於 {out_dir}）", flush=True)
        if progress_cb:
            try:
                progress_cb(
                    {
                        "stage": "done",
                        "done": True,
                        "session": session,
                        "gpu": gpu,
                        "_local_synced_jobs": list(done_ids),
                        "_local_results": prev_results,
                        "log": ["全部任務皆已完成且指紋相符，免啟動 GPU"],
                    }
                )
            except Exception:
                pass
        return prev_results

    if done_ids:
        print(f"⏭️ 略過已完成且指紋相符的任務：{', '.join(done_ids)}", flush=True)

    t0 = time.monotonic()
    deadline = t0 + max_minutes * 60
    wall_deadline = time.time() + max_minutes * 60
    run_id = uuid.uuid4().hex[:12]

    def _emit(payload: dict[str, Any]):
        if not progress_cb:
            return
        try:
            snap = dict(payload)
            snap.setdefault("run_id", run_id)
            snap.setdefault("session", session)
            snap.setdefault("gpu", gpu)
            snap.setdefault("_local_synced_jobs", sorted(local_synced))
            snap.setdefault("_local_results", json.loads(json.dumps(results, ensure_ascii=False)))
            progress_cb(snap)
        except Exception:
            pass

    abort_reason: str | None = None
    stopped: RunStopped | None = None
    remote_attempted = False
    remote_conflict = False
    cancel_pending = False
    cancel_sent = False
    cancel_failed = False
    endpoint: str | None = None
    attached = False

    def _run_lease(info: dict) -> bool:
        return info.get("account_scope") == account and (
            (info.get("runtime_endpoint") == endpoint and info.get("active_run_id") == run_id)
            or (info.get("allocation_id") == run_id and info.get("created_by") == "pending")
        )

    def _check_abort():
        nonlocal abort_reason, cancel_pending, stopped
        if not should_abort:
            return
        sig = should_abort()
        if not sig:
            return
        if sig == "stop":
            abort_reason = "stop"
            stopped = RunStopped("使用者要求中斷任務並關閉 Colab GPU (stopped by user)")
            raise stopped
        if remote_attempted:
            cancel_pending = True
            return
        abort_reason = "cancel"
        raise RunCancelled("本地已取消，尚未啟動遠端工作 (cancelled by user)")

    def _check_active():
        _check_abort()
        if time.monotonic() >= deadline:
            raise RuntimeError(f"超過 {max_minutes} 分鐘上限，中止以保護運算單元")
        if C.account_scope() != account:
            raise RuntimeError("Colab 登入身分已變更，請重新確認目標 Session")
        if endpoint and C.runtime_endpoint(session, deadline=deadline) != endpoint:
            raise RuntimeError(f"Session {session} 的遠端識別已變更，請先確認目標")
        if attached and not _run_lease(C.keep_info(session) or {}):
            raise C.KeepChanged(f"Session {session} 的執行紀錄已變更或正在關機")

    local_synced: set[str] = set(done_ids)
    results: dict[str, Any] = {
        "session": session,
        "gpu": gpu,
        "run_id": run_id,
        "model": m.get("model", DEFAULT_MODEL),
        "jobs": dict((prev_results.get("jobs") or {}) if not force else {}),
    }

    _check_abort()
    account = account or C.account_scope()
    if not account or C.account_scope() != account:
        _fail("無法確認 Colab 登入身分，請重新確認目標 Session", exit_on_error)
    cu_before, _ = C.usage(account=account, deadline=deadline)
    _check_active()
    alive, listing = C.sessions_checked(account=account, deadline=deadline)
    _check_active()
    assignments = C.listed_assignments(listing) if alive is not None else None
    if assignments is None:
        _fail("無法確認開啟中的 Colab Session，請稍後再試", exit_on_error)
    active = {name: ep for ep, name in assignments.items() if name != "?"}

    allocation_reserved = False
    allocation_attempted = False
    created = False
    owned = False

    got: set[str] = set()
    expected_todo_ids = {j["id"] for j in todo}
    finished = False

    try:
        _check_active()
        if auto_reuse and session is None:
            registered = C.owned_sessions()
            session = next((name for name in registered if C.is_owned_session(
                name, account=account, assignments=assignments, deadline=deadline,
            ) and not (C.keep_info(name) or {}).get("release_pending")), None)
            if session is None and (
                "?" in assignments.values() or active.keys() & set(registered) or C.has_pending_allocation(account)
            ):
                _fail("有尚待確認的 Colab Session，請先檢查目標或明確選擇新開", exit_on_error)
        reuse = session in active
        named = session is not None
        session = session or C.new_session_name()
        results["session"] = session
        record = C.keep_info(session)
        keep = record or {}
        if record is None and C.keep_file(session).exists():
            _fail(f"無法讀取 Session {session} 的本機識別紀錄", exit_on_error)
        if keep.get("account_scope", account) != account:
            _fail(f"Session {session} 的記錄屬於另一個 Colab 登入身分", exit_on_error)
        if keep.get("created_by") == "pending" or keep.get("release_pending"):
            _fail(f"Session {session} 的配置或關機仍待確認，請先檢查 Colab runtime 狀態", exit_on_error)
        if keep.get("active_run_id") and keep.get("created_by") == "owned":
            _fail(f"Session {session} 已有批次的執行紀錄，請先等待批次完成或回收", exit_on_error)

        if reuse:
            endpoint = active[session]
            if keep.get("runtime_endpoint") not in (None, endpoint) or C.runtime_endpoint(session, deadline=deadline) != endpoint:
                _fail(f"Session {session} 的遠端識別已變更，請先確認目標", exit_on_error)
            owned = C.is_owned_session(session, account=account, assignments=assignments, deadline=deadline)
            st_check = C.poll_remote_status(session, account=account, endpoint=endpoint, deadline=deadline)
            if not isinstance(st_check, dict) or type(st_check.get("alive")) is not bool:
                _fail(f"無法確認 Session {session} 的遠端工作狀態", exit_on_error)
            if st_check.get("stage") == "cancel_failed":
                _fail(f"Session {session} 仍有待確認的取消結果，請先檢查遠端任務", exit_on_error)
            if st_check.get("alive"):
                _fail(
                    f"⚠️ Session {session} 上已有正在執行的批次任務，請先等待完成或執行 `uv run cstudio cancel --session {session}`",
                    exit_on_error,
                )
            print(f"♻️ 沿用開啟中的 Colab Session：{session}（省去安裝與重複下載模型）", flush=True)
            _emit({"stage": "reusing_vm", "done": False, "log": [f"沿用開啟中的 Colab Session：{session}"]})
            _check_active()
            C.set_keep(
                session, None, owner="owned" if owned else "external", expected=record,
                extra={"busy_until": wall_deadline if owned else None, "runtime_endpoint": endpoint, "active_run_id": run_id,
                       "account_scope": account,
                       "bundle_hash": keep.get("bundle_hash") if keep.get("runtime_endpoint") == endpoint else None},
            )
            attached = True
            if owned and endpoint:
                C.start_watchdog(session, require_ready=True, deadline=deadline)
        else:
            if reuse_only or C.keep_file(session).exists() or (named and "?" in assignments.values()):
                _fail(f"無法確認 Session {session} 可沿用或安全配置，請先檢查 Colab runtime 狀態", exit_on_error)
            print(f"🚀 在 Google Colab 開啟 {gpu}：{session}", flush=True)
            _emit({"stage": "booting_vm", "done": False, "log": [f"正在 Google Colab 分配並啟動 {gpu} ({session})..."]})
            _check_active()
            if C.runtime_endpoint(session, deadline=deadline):
                _fail(f"Session {session} 已有本機配置紀錄，請先確認 Colab runtime 狀態", exit_on_error)
            reserved = C.set_keep(
                session, None, owner="pending",
                extra={"busy_until": wall_deadline + C.WATCHDOG_TICK, "runtime_endpoint": None,
                       "bundle_hash": None, "account_scope": account, "allocation_id": run_id,
                       "allocation_baseline": sorted(assignments)},
                reserve=True,
            )
            allocation_reserved = True
            C.start_watchdog(session, require_ready=True, deadline=deadline)
            _check_active()
            allocation_attempted = True
            endpoint = C.new_runtime(
                session, gpu, retries=assign_retries, wait=assign_wait,
                check_abort=_check_active, deadline=deadline, account=account,
            )
            attached = created = owned = True
            C.set_keep(session, None, owner="owned", expected=reserved,
                       extra={"busy_until": wall_deadline, "runtime_endpoint": endpoint, "active_run_id": run_id})

        _check_active()
        target = dict(account=account, endpoint=endpoint, deadline=deadline)
        short = dict(timeout=45, tries=6, **target)
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

        sources = {rel: (REMOTE / rel).read_text(encoding="utf-8") for rel in REMOTE_CODE_FILES}
        bundle_hash = bundle_digest(sources)
        k_info = C.keep_info(session) or {}
        need_code = created or k_info.get("bundle_hash") != bundle_hash

        files = sorted({f for j in todo for f in (j.get("inputs") or [])})
        input_shas = {f: hashlib.sha256(confined_path(base, f).read_bytes()).hexdigest() for f in files}
        remote_run = f"{C.REMOTE_ROOT}/runs/{run_id}"

        def request_preparation(code: str) -> dict:
            nonlocal remote_attempted, remote_conflict
            remote_attempted = True
            reply = _prepare_result(C.remote_exec(session, code, **short), run_id, bundle_hash, input_shas)
            remote_attempted = reply["state"] in ("running", "finished")
            if reply["state"] == "busy":
                remote_conflict = True
                if reply.get("reason") == "cancel_failed":
                    raise RuntimeError(f"Session {session} 仍有待確認的取消結果")
                raise RuntimeError(f"Session {session} 已有執行中的批次，遠端拒絕本批啟動")
            return reply

        _emit({"stage": "syncing_code", "done": False, "log": [f"正在同步遠端程式與任務設定至 {session}..."]})
        _check_active()
        prep = request_preparation(bundle_and_launch_code(sources if need_code else None, bundle_hash, manifest_json, input_shas))
        if prep["state"] == "bundle_required" and not need_code:
            _check_active()
            prep = request_preparation(bundle_and_launch_code(sources, bundle_hash, manifest_json, input_shas))
        if prep["state"] == "bundle_required":
            raise RuntimeError("遠端未確認批次所需的程式版本")
        _check_active()
        current = C.keep_info(session) or {}
        if _run_lease(current) and current.get("bundle_hash") != bundle_hash:
            try:
                C.set_keep(session, current.get("expires"), expected=current, extra={"bundle_hash": bundle_hash})
            except C.KeepChanged:
                pass  # The bundle cache can be rebuilt after a concurrent keep update.

        for f in prep["missing_inputs"]:
            _check_active()
            print(f"⬆️ 上傳參考圖：{f}", flush=True)
            _emit({"stage": "uploading_inputs", "done": False, "log": [f"正在上傳參考圖片：{f}"]})
            C.colab("upload", "-s", session, confined_path(base, f), f"{remote_run}/uploads/{f}", timeout=600, **target)

        _check_active()
        if prep["state"] == "prepared":
            launch = C.vm_program("PREP", "launch_run", run_id, bundle_hash, list(REMOTE_CODE_FILES), include_start=True)
            prep = request_preparation(launch)
            if not remote_attempted:
                raise RuntimeError("遠端尚未啟動本批，程式版本或參考圖片未通過驗證")
        results["bundle_hash"] = bundle_hash
        _emit({"stage": "starting", "done": False, "log": ["已確認遠端接受本批次"]})

        shown_log: set[str] = set()
        last_ok = time.monotonic()

        while True:
            time.sleep(C.remaining_time(4, deadline))
            _check_active()

            if cancel_pending and not cancel_sent:
                _emit({"stage": "cancelling", "done": False})
                try:
                    receipt = C.request_cancel(session, run_id, allow_start=True, **target)
                except (OSError, RuntimeError, ValueError):
                    receipt = {"state": "unknown"}
                if receipt["state"] == "unsupported":
                    raise RuntimeError("此遠端批次不支援指定任務取消，無法確認已停止")
                cancel_sent = receipt["state"] in {"pending", "confirmed", "finished", "failed"}

            st = C.poll_remote_status(session, run_id=run_id, **target)
            if not isinstance(st, dict) or st.get("run_id") != run_id or st.get("bundle_hash") != bundle_hash:
                if time.monotonic() - last_ok > 15 * 60:
                    raise RuntimeError("連續 15 分鐘無法讀取 Colab 狀態，連線可能已中斷")
                print("…等待 Colab 回應中", flush=True)
                continue

            st = redact_diagnostics(st)
            if isinstance(st.get("jobs"), dict):
                st["jobs"] = {
                    jid: redact_diagnostics(info) if isinstance(info, dict) else info
                    for jid, info in st["jobs"].items()
                }
            if st.get("stage") == "cancelled" and st.get("cancel_protocol") != 1:
                raise RuntimeError("收到無法驗證的遠端取消結果")
            if st.get("active_run_id") not in (None, run_id) or st.get("legacy_alive"):
                remote_conflict = True

            last_ok = time.monotonic()
            for line in st.get("log") or []:
                if line not in shown_log:
                    shown_log.add(line)
                    print(f"  [Colab] {line}", flush=True)

            # If remote reports done=True, hold stage as "syncing_outputs" until local download + SHA256 verification completes
            st_for_ui = dict(st)
            if cancel_pending and not st.get("done"):
                st_for_ui["stage"] = "cancelling"
            elif st_for_ui.get("done") and st_for_ui.get("stage") != "cancelled" and not st_for_ui.get("error"):
                st_for_ui["stage"] = "syncing_outputs"
            _emit(st_for_ui)

            # Pull newly completed images immediately via atomic temporary file + sha256 verification
            r_jobs = st.get("jobs") or {}
            if not isinstance(r_jobs, dict):
                raise RuntimeError("遠端成果清單格式錯誤")
            synced_any = False
            for jid, jinfo in r_jobs.items():
                if jid not in expected_todo_ids or jid in got or not isinstance(jinfo, dict):
                    continue
                if jinfo.get("status") == "done":
                    expected_file = f"out/{jid}.png"
                    if jinfo.get("file") != expected_file:
                        if st.get("done"):
                            results["jobs"][jid] = {"status": "sync_error", "run_id": run_id, "error": "遠端成果路徑與當批任務不符"}
                            save_results(res_file, results)
                            got.add(jid)
                        continue
                    tmp_pull = confined_path(base, f"out/.{jid}_{run_id}.png.tmp", missing_ok=True)
                    final_dst = confined_path(base, expected_file, missing_ok=True)
                    max_pull_tries = 3 if st.get("done") else 1
                    pull_err = None
                    for pull_try in range(max_pull_tries):
                        _check_active()
                        try:
                            C.colab("download", "-s", session, f"{remote_run}/{expected_file}", tmp_pull,
                                    timeout=600, **target)
                            pulled = confined_path(base, f"out/{tmp_pull.name}")
                            if pulled.stat().st_size == 0:
                                raise RuntimeError("下載的暫存檔為空")
                            exp_sha = jinfo.get("sha256")
                            got_sha = hashlib.sha256(pulled.read_bytes()).hexdigest()
                            if exp_sha and got_sha != exp_sha:
                                raise RuntimeError(f"SHA256 校驗不符 (expected {exp_sha[:10]}, got {got_sha[:10]})")
                            pulled.replace(final_dst)
                            rec = dict(jinfo, run_id=run_id, fingerprint=fps.get(jid), local_synced=True)
                            results["jobs"][jid] = rec
                            save_results(res_file, results)
                            got.add(jid)
                            local_synced.add(jid)
                            synced_any = True
                            print(f"⬇️ 已同步回本地：{final_dst} ({final_dst.stat().st_size / 1024:.1f} KB)", flush=True)
                            if rec.get("refined_prompt"):
                                print(f"   ✨ Prompt Enhancer 擴寫結果：{redact(rec['refined_prompt'])[:160]}...", flush=True)
                            pull_err = None
                            break
                        except C.ColabDeadline:
                            tmp_pull.unlink(missing_ok=True)
                            raise
                        except Exception as e:
                            pull_err = redact(str(e))
                            tmp_pull.unlink(missing_ok=True)
                            if pull_try + 1 < max_pull_tries:
                                print(f"⚠️ 拉回 {jid}.png 失敗，重試中 ({pull_try + 1}/{max_pull_tries})：{pull_err}", flush=True)
                                time.sleep(C.remaining_time(2, deadline))
                            else:
                                print(f"⚠️ 拉回 {jid}.png 失敗：{pull_err}", flush=True)
                    if pull_err and st.get("done"):
                        results["jobs"][jid] = dict(jinfo, run_id=run_id, status="sync_error", error=f"遠端已生成但下載回本地失敗：{pull_err}")
                        save_results(res_file, results)
                        got.add(jid)
                        synced_any = True
                elif jinfo.get("status") == "error":
                    results["jobs"][jid] = dict(jinfo, run_id=run_id)
                    save_results(res_file, results)
                    got.add(jid)
                    synced_any = True
                    print(f"❌ 任務 {jid} 失敗：{jinfo.get('error')}", flush=True)

            if synced_any and not st.get("done"):
                _emit(st_for_ui)

            if st.get("done"):
                _check_abort()
                stage = st.get("stage")
                if stage == "cancelled":
                    abort_reason = "cancel"
                    _emit(st)
                    raise RunCancelled("已確認遠端批次取消 (cancelled by user)")
                cancel_failed = stage == "cancel_failed"
                if st.get("error") and stage in ("error", "crashed", "cancel_failed"):
                    results["error"] = st["error"]
                finished = True
                _emit(st)
                break

    except BaseException as exc:
        if "error" not in results:
            results["error"] = redact(str(exc))
        raise
    finally:
        results["minutes"] = round((time.monotonic() - t0) / 60, 2)
        if attached:
            owner = "owned" if owned else "external"
            cleanup_deadline = time.monotonic() + C.RECOVERY_TIMEOUT
            try:
                current = C.keep_info(session) or {}
                same_keep = _run_lease(current)
                release = abort_reason == "stop" or (
                    not remote_conflict and abort_reason != "cancel" and (
                        (stop_after and (finished or (owned and not remote_attempted)))
                        or (not finished and (created or (owned and time.monotonic() >= deadline)))
                    )
                )
                if release:
                    print(f"🛑 關閉 Colab GPU：{session}", flush=True)
                    outcome = (C.release_runtime(session, owner, endpoint, account=account, deadline=cleanup_deadline)
                               if same_keep or not current else C.ReleaseState.FAILED)
                    results["shutdown_state"] = outcome
                    if stopped is not None:
                        stopped.outcome = outcome
                    if outcome != C.ReleaseState.CONFIRMED:
                        results["shutdown_pending"] = outcome == C.ReleaseState.PENDING
                        results.setdefault("error", f"尚未確認 {session} 已關閉，請檢查 Colab runtime 狀態")
                        print(f"⚠️ 尚未確認 {session} 已關閉", flush=True)
                elif not same_keep:
                    print(f"⚠️ Session {session} 的本機紀錄已變更，請檢查 Colab runtime 狀態", flush=True)
                elif cancel_failed or (cancel_pending and not finished and abort_reason != "cancel"):
                    print(f"⚠️ 尚未確認 {session} 的批次已取消，請檢查遠端狀態", flush=True)
                elif owned and remote_attempted and not finished and abort_reason != "cancel":
                    print(f"⚠️ 批次狀態待確認，保留 {session} 的原回收期限", flush=True)
                elif finished or owned or abort_reason == "cancel":
                    keep_mins = max(30.0, float(idle_minutes)) if abort_reason == "cancel" else idle_minutes
                    try:
                        for _ in range(2):
                            current = C.keep_info(session) or {}
                            if not _run_lease(current):
                                break
                            expires = time.time() + keep_mins * 60
                            previous = current.get("expires")
                            if isinstance(previous, (int, float)):
                                expires = max(expires, previous)
                            try:
                                C.set_keep(session, expires, owner=owner, expected=current,
                                           extra={"busy_until": None, "active_run_id": None})
                            except C.KeepChanged:
                                continue
                            C.start_watchdog(session)
                            print(f"⏸️ {session} 的待命期限已更新", flush=True)
                            break
                    except OSError:
                        if created:
                            outcome = C.release_runtime(session, owner, endpoint, account=account, deadline=cleanup_deadline)
                            results["shutdown_state"] = outcome
                            results["shutdown_pending"] = outcome == C.ReleaseState.PENDING
                        raise
                else:
                    C.drop_keep(session, expected=current)
                    print(f"⚠️ 批次未完成，外部 Session {session} 仍在運行。", flush=True)
            finally:
                cu_after, _ = C.usage(account=account, deadline=cleanup_deadline)
                cu_used = round(cu_before - cu_after, 2) if (cu_before is not None and cu_after is not None) else None
                results.update(cu_before=cu_before, cu_after=cu_after, cu_used=cu_used)
                save_results(res_file, results)
                if cu_used is not None:
                    print(f"💳 本批預估消耗 {cu_used} 運算單元（{cu_before} → {cu_after}；Colab 用量可能延遲更新）", flush=True)
        elif allocation_reserved:
            current = C.keep_info(session)
            if not _run_lease(current or {}):
                current = None
            if current and allocation_attempted:
                try:
                    review_at = time.time() + 30
                    C.set_keep(session, None, owner="pending", expected=current, extra={"busy_until": review_at})
                except (OSError, C.KeepChanged):
                    pass  # The reservation was already persisted before allocation.
                print(f"⚠️ 尚未確認 {session} 的配置結果。請檢查 Colab runtime，含未命名的機器。", flush=True)
            elif current:
                C.drop_keep(session, expected=current)

    expected_ids = [j["id"] for j in m["jobs"]]
    bad = [
        jid
        for jid in expected_ids
        if (results["jobs"].get(jid) or {}).get("status") != "done" or not (out_dir / f"{jid}.png").is_file()
    ]
    print(f"✅ 批次結束：成功 {len(expected_ids) - len(bad)}/{len(expected_ids)}，耗時 {results['minutes']} 分鐘 → {out_dir}", flush=True)
    if bad or results.get("error"):
        if not results.get("error"):
            err_details = []
            for jid in bad:
                j_err = (results["jobs"].get(jid) or {}).get("error")
                err_details.append(f"{jid} ({j_err})" if j_err else jid)
            results["error"] = f"部分任務未成功完成：{', '.join(err_details)}"
            save_results(res_file, results)
        if exit_on_error:
            sys.exit(1)
    return results
