# -*- coding: utf-8 -*-
"""Batch validation, fingerprinting, planning, and Colab execution/sync engine."""
from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from filelock import Timeout as LockTimeout

from remote.job_paths import JOB_ID, RUN_ID, bundle_digest, confined_path, file_sha256, json_digest
from remote.manifest import normalize_manifest, normalize_quant, parse_manifest

from . import colab_client as C
from .redact import redact, redact_diagnostics
from .results import (
    WORKSPACE_LOCK_TIMEOUT, RunOutputs, SupersededResult, WorkspaceResults, done_revision, execution_digest,
    is_sha256, matches_image, read_object, save_results, verify_receipt,
)

ROOT = Path(__file__).resolve().parent.parent
REMOTE = ROOT / "remote"
OUTPUT_RECOVERY_TIMEOUT = 120
sys.path.insert(0, str(REMOTE))
from models import DEFAULT_MODEL, get_model_plugin  # noqa: E402


def validate_manifest(manifest: dict[str, Any], base_dir: Path, cli_quant: str | None = None) -> list[str]:
    _, problems = normalize_manifest(manifest, base_dir, cli_quant)
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
        data = parse_manifest(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        _fail(f"manifest JSON 格式錯誤（{p}）：{redact(str(exc))}", exit_on_error)
    m, problems = normalize_manifest(data, base, cli_quant)
    if problems:
        _fail("manifest 驗證未通過，沒有開機：\n- " + redact("\n- ".join(problems)), exit_on_error)
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
    *,
    input_shas: dict[str, str] | None = None,
) -> str:
    """Compute the original graph fingerprint for existing results."""
    plugin = get_model_plugin(manifest.get("model"))
    graph = plugin.build_qwen21_graph(job, manifest=manifest, cli_quant=cli_quant)

    if input_shas is None:
        input_shas = _reference_hashes([job], base_dir)
    input_hashes = [{"path": rel, "sha256": input_shas[rel]} for rel in job.get("inputs") or []]

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
    if plugin.uses_separate_edit_pe(job, manifest):
        payload["pe_execution"] = "edit_prepass_v1"
    elif plugin.uses_separate_transparent_t2i_pe(job, manifest):
        payload["pe_execution"] = "transparent_t2i_prepass_v2"
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reference_hashes(jobs: list[dict], base: Path) -> dict[str, str]:
    hashes = {}
    for rel in sorted({ref for job in jobs for ref in job.get("inputs") or []}):
        path = confined_path(base, rel, missing_ok=True)
        hashes[rel] = file_sha256(path) if path.is_file() else "missing"
    return hashes


def _stale_output(jid, image, record, fingerprint, legacy, *, use_fingerprint, pin_versions, pin_models):
    has_integrity = "sha256" in record or "bytes" in record
    if image.stat().st_size == 0:
        return "本機成果是空檔"
    if has_integrity:
        try:
            verify_receipt(jid, record)
        except ValueError:
            return "本機成果的大小或 SHA-256 紀錄無效"
        if not matches_image(image, record):
            return "本機成果的大小或 SHA-256 已變更"
    if not use_fingerprint:
        return None
    if not has_integrity:
        return "舊成果缺少大小與 SHA-256 紀錄"
    version = record.get("fingerprint_version")
    if version is None and record.get("fingerprint") == legacy and pin_versions and pin_models:
        return None
    if version != 2 or record.get("fingerprint") != fingerprint:
        return "生成流程、參考圖片、版本或執行設定已變更"
    execution = record.get("execution")
    try:
        digest = execution_digest(fingerprint, execution, pin_versions=pin_versions, pin_models=pin_models)
    except ValueError:
        return "本機成果的執行環境紀錄無效"
    if digest != record.get("execution_fingerprint"):
        return "本機成果的執行環境指紋不符"
    if not (pin_versions and pin_models and execution["comfy"]["verified"]):
        return "執行環境未完整釘選或驗證，離線無法確認相容"
    if not {"comfyui_version", "python_version", "pytorch_version"} <= execution["comfy"]["versions"].keys():
        return "ComfyUI 未回報完整版本，離線無法確認相容"
    return None


def classify_jobs(
    manifest: dict[str, Any],
    base_dir: Path,
    cli_quant: str | None = None,
    use_fingerprint: bool = True,
    *,
    gpu: str = "L4",
    pin_versions: bool = True,
    pin_models: bool = True,
    bundle_hash: str | None = None,
    input_shas: dict[str, str] | None = None,
    versions: dict | None = None,
    prior_jobs: dict | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], list[str], list[tuple[str, str]]]:
    """Return (fps, todo_jobs, done_ids, stale_list[(jid, reason)])."""
    if prior_jobs is None:
        prior_jobs = read_object(WorkspaceResults(base_dir).path).get("jobs", {})
    if not isinstance(prior_jobs, dict):
        raise ValueError("results.json 的成果清單格式錯誤")

    if versions is None:
        versions = load_versions()
    pin_models = pin_models and pin_versions
    run_settings = {"gpu": gpu, "pin_versions": pin_versions, "pin_models": pin_models,
                    "bundle_hash": bundle_hash or compute_remote_bundle_hash()}
    if input_shas is None:
        input_shas = _reference_hashes(manifest["jobs"], base_dir)
    fps: dict[str, str] = {}
    todo: list[dict[str, Any]] = []
    done_ids: list[str] = []
    stale: list[tuple[str, str]] = []

    for j in manifest.get("jobs") or []:
        jid = j["id"]
        legacy = compute_job_fingerprint(j, manifest, base_dir, cli_quant, versions, input_shas=input_shas)
        fp = json_digest({"schema": 2, "graph_fingerprint": legacy, **run_settings})
        fps[jid] = fp
        out_img = confined_path(base_dir, f"out/{jid}.png", missing_ok=True)
        if not out_img.is_file():
            todo.append(j)
            continue
        prev = prior_jobs.get(jid)
        reason = _stale_output(jid, out_img, prev if isinstance(prev, dict) else {}, fp, legacy,
                               use_fingerprint=use_fingerprint, pin_versions=pin_versions, pin_models=pin_models)
        if reason is None:
            done_ids.append(jid)
        else:
            stale.append((jid, reason))
            todo.append(j)

    return fps, todo, done_ids, stale


def _snapshot_references(base: Path, destination: Path, hashes: dict[str, str]):
    for rel, digest in hashes.items():
        staged = confined_path(destination, rel, missing_ok=True)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(confined_path(base, rel), staged)
        if file_sha256(staged) != digest:
            raise RuntimeError(f"參考圖在批次準備期間已變更：{rel}，請重新執行")


def _legacy_outputs(results: dict, done_ids: list[str]) -> list[str]:
    records = results.get("jobs", {})
    return [
        jid for jid in done_ids if not isinstance(records.get(jid), dict)
        or records[jid].get("fingerprint_version") != 2
    ]


def plan_batch(
    manifest_path: str | Path, gpu: str = "L4", cli_quant: str | None = None,
    *, pin_versions: bool = True, pin_models: bool = True,
) -> dict[str, Any]:
    m, base = load_manifest(manifest_path, cli_quant=cli_quant)
    cli_quant = normalize_quant(cli_quant)
    plugin = get_model_plugin(m.get("model"))
    try:
        _, todo, done_ids, stale = classify_jobs(m, base, cli_quant=cli_quant, gpu=gpu,
                                                pin_versions=pin_versions, pin_models=pin_models)
        legacy = _legacy_outputs(read_object(confined_path(base, "out/results.json", missing_ok=True)), done_ids)
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
    stale_reasons = dict(stale)
    for j in m["jobs"]:
        jid = j["id"]
        w, h = plugin.resolve_dimensions(j, m)
        q = (cli_quant or j.get("quant") or m.get("quant") or plugin.DEFAULT_QUANT).upper()
        ref = "✨PE擴寫" if j.get("refine_prompt", m.get("refine_prompt", False)) else "直出"
        rgba = " [RGBA透明]" if j.get("transparent") else ""
        st = "🆕 待生成"
        if jid in legacy:
            st = "✅ 舊成果已通過圖片驗證"
        elif jid in done_ids:
            st = "✅ 已完成"
        elif jid in stale_reasons:
            st = f"⚠️ 過期（{stale_reasons[jid]}）"
        print(f"     - {jid}: {st} | mode={j.get('mode', 't2i')}{rgba} | quant={q} | {w}x{h} | {ref}", flush=True)

    return {
        "total": len(m["jobs"]),
        "done": done_ids,
        "legacy": legacy,
        "todo": [j["id"] for j in todo],
        "stale": stale,
        "models": model_list,
        "total_gb": round(total_bytes / 1e9, 2),
    }


def recover_outputs(manifest_path: str | Path, run_id: str) -> dict:
    """Resume local publication and fetch completed jobs from the saved runtime."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("不合法的批次 ID")
    base = Path(manifest_path).absolute().parent.resolve()
    path = confined_path(base, f"out/.runs/{run_id}/request.json", missing_ok=True)
    request = read_object(path)
    if not request:
        raise RuntimeError("找不到此批次的補抓紀錄，請核對工作目錄與批次 ID")
    session, account, endpoint = (request.get(key) for key in ("session", "account_scope", "runtime_endpoint"))
    jobs = request.get("jobs")
    if (
        type(request.get("schema")) is not int or request["schema"] not in (1, 2) or request.get("run_id") != run_id
        or (request["schema"] == 2 and (
            type(request.get("pin_versions")) is not bool or type(request.get("pin_models")) is not bool
            or (request["pin_models"] and not request["pin_versions"])
        ))
        or not is_sha256(request.get("bundle_hash"))
        or not all(isinstance(value, str) and value for value in (session, account, endpoint))
        or C.sanitize_session_name(session) != session or not isinstance(jobs, dict) or not jobs
        or any(
            not JOB_ID.fullmatch(jid) or not isinstance(job, dict) or not is_sha256(job.get("fingerprint"))
            or (job.get("previous") is not None and not is_sha256(job["previous"]))
            or type(job.get("archive")) is not bool
            for jid, job in jobs.items()
        )
    ):
        raise ValueError("批次補抓紀錄格式或識別不符")
    outputs = RunOutputs(base, run_id, request)
    try:
        with outputs.lock():
            return _recover_outputs(outputs, session, account, endpoint)
    except LockTimeout:
        raise RuntimeError("此批次正在本機執行或補抓，請稍後重試") from None


def _recover_outputs(outputs: RunOutputs, session: str, account: str, endpoint: str) -> dict:
    report = {"synced": [], "already_synced": [], "pending": [], "unavailable": [], "errors": {}, "superseded": []}
    deadline = time.monotonic() + OUTPUT_RECOVERY_TIMEOUT

    def check_local():
        C.remaining_time(OUTPUT_RECOVERY_TIMEOUT, deadline)

    def check_remote():
        check_local()
        C.check_recovery_target(session, account, endpoint, deadline=deadline)

    def download(file, dest):
        C.colab("download", "-s", session, f"{C.REMOTE_ROOT}/runs/{outputs.run_id}/{file}", dest, **target)

    remaining = []
    for jid in outputs.request["jobs"]:
        try:
            record, committed = outputs.publish(
                jid, check_local, timeout=C.remaining_time(WORKSPACE_LOCK_TIMEOUT, deadline),
            )
            if record is not None:
                report["synced" if committed else "already_synced"].append(jid)
            else:
                remaining.append(jid)
        except SupersededResult:
            report["superseded"].append(jid)
        except (OSError, RuntimeError, ValueError) as exc:
            report["errors"][jid] = redact(str(exc))
    if not remaining:
        return report

    target = dict(account=account, endpoint=endpoint, deadline=deadline)
    try:
        C.verify_recovery_runtime(session, **target)
        st = C.poll_remote_status(session, run_id=outputs.run_id, **target)
        check_remote()
        if not isinstance(st, dict):
            raise RuntimeError("無法讀取原批次的遠端狀態，請稍後重試")
        if st.get("run_id") == outputs.run_id and st.get("stage") == "not started" and not st.get("bundle_hash"):
            raise C.RuntimeUnavailable("原 Colab runtime 缺少這個批次的遠端紀錄")
        if st.get("run_id") != outputs.run_id or st.get("bundle_hash") != outputs.request["bundle_hash"]:
            raise RuntimeError("遠端狀態的批次 ID 或程式版本不符")
        remote_jobs = st.get("jobs")
        if not isinstance(remote_jobs, dict):
            raise RuntimeError("遠端成果清單格式錯誤")
    except (OSError, RuntimeError, ValueError) as exc:
        report["remote_error"] = redact(str(exc))
        report["unavailable" if isinstance(exc, C.RuntimeUnavailable) else "pending"] = remaining
        return report

    for jid in remaining:
        info = remote_jobs.get(jid)
        if isinstance(info, dict) and info.get("status") == "done":
            try:
                outputs.pull(jid, redact_diagnostics(info), download, check_remote)
                record, _ = outputs.publish(
                    jid, check_remote, timeout=C.remaining_time(WORKSPACE_LOCK_TIMEOUT, deadline),
                )
                if record is None:
                    raise RuntimeError("找不到已驗證的暫存成果")
            except SupersededResult:
                report["superseded"].append(jid)
            except (OSError, RuntimeError, ValueError) as exc:
                report["errors"][jid] = redact(str(exc))
            else:
                report["synced"].append(jid)
        elif st.get("done"):
            report["errors"][jid] = "原批次已結束，沒有此任務已完成的成果"
        else:
            report["pending"].append(jid)
    return report


REMOTE_CODE_FILES = (
    "__init__.py",
    "cancel.py",
    "job_paths.py",
    "manifest.py",
    "remote_runner.py",
    "versions.json",
    "models/__init__.py",
    "models/qwen_image_21.py",
    "models/qwen_prompts.py",
)


def compute_remote_bundle_hash() -> str:
    return bundle_digest(_remote_sources())


def _remote_sources() -> dict[str, str]:
    return {rel: (REMOTE / rel).read_text(encoding="utf-8") for rel in REMOTE_CODE_FILES}


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
    expected_endpoint: str | None = None,
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
    m, base = load_manifest(manifest_path, cli_quant=cli_quant, exit_on_error=exit_on_error)
    cli_quant = normalize_quant(cli_quant)
    pin_versions = bool(pin_versions)
    pin_models = pin_versions and bool(pin_models)
    workspace = WorkspaceResults(base)
    try:
        out_dir = confined_path(base, "out", directory=True, missing_ok=True)
        sources = _remote_sources()
        bundle_hash = bundle_digest(sources)
        all_input_shas = _reference_hashes(m["jobs"], base)
        out_dir.mkdir(exist_ok=True)
        with workspace.transaction() as prev_results:
            fps, todo, done_ids, stale = classify_jobs(
                m, base, cli_quant=cli_quant, use_fingerprint=use_fingerprint, gpu=gpu,
                pin_versions=pin_versions, pin_models=pin_models, bundle_hash=bundle_hash,
                versions=json.loads(sources["versions.json"]), input_shas=all_input_shas,
                prior_jobs=prev_results["jobs"],
            )
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"無法使用任務檔案：{exc}", exit_on_error)

    if force:
        todo = list(m["jobs"])
        done_ids = []
        stale = []
    elif stale and not remake_stale:
        msg = "發現已存在的輸出與目前設定不符（過期）：\n" + "\n".join(f"  - {jid}: {reason}" for jid, reason in stale)
        msg += "\n👉 若要重做過期項目（舊檔會移至 out/_stale/），請加上 `--remake-stale`；若要全部重做請加 `--force`。"
        _fail(msg, exit_on_error)

    if use_fingerprint and (legacy := _legacy_outputs(prev_results, done_ids)):
        print(f"📄 沿用已通過圖片驗證的舊成果，原執行環境沒有紀錄：{', '.join(legacy)}", flush=True)
    if not todo:
        print(f"✅ 全部 {len(m['jobs'])} 個任務沿用本機成果，無需啟動 Colab GPU（{out_dir}）", flush=True)
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
                        "log": ["沿用全部本機成果，免啟動 GPU"],
                    }
                )
            except Exception:
                pass
        return prev_results

    if not C.token_ok():
        _fail("尚未登入 Google Colab：請先執行 `uv run cstudio login url`", exit_on_error)
    if done_ids:
        print(f"⏭️ 沿用已有成果的任務：{', '.join(done_ids)}", flush=True)

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
            snap["session"] = session
            snap.setdefault("gpu", gpu)
            snap["_runtime_endpoint"] = endpoint
            snap["_listed_hardware"] = listed_hardware
            snap["_runtime_attached"] = attached
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
    listed_hardware: str | None = None
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

    def _check_active(limit: float = deadline):
        _check_abort()
        if time.monotonic() >= deadline:
            raise RuntimeError(f"超過 {max_minutes} 分鐘上限，中止以保護運算單元")
        if C.account_scope() != account:
            raise RuntimeError("Colab 登入身分已變更，請重新確認目標 Session")
        if endpoint and C.runtime_endpoint(session, deadline=limit) != endpoint:
            raise RuntimeError(f"Session {session} 的遠端識別已變更，請先確認目標")
        if attached and not _run_lease(C.keep_info(session) or {}):
            raise C.KeepChanged(f"Session {session} 的執行紀錄已變更或正在關機")

    local_synced: set[str] = set(done_ids)
    results: dict[str, Any] = {
        "session": session,
        "gpu": gpu,
        "run_id": run_id,
        "model": m.get("model", DEFAULT_MODEL),
        "jobs": dict(prev_results.get("jobs") or {}),
    }

    def _record_job_error(jid: str, info: dict, status: str, message: str):
        rec = dict(info, run_id=run_id, bundle_hash=results["bundle_hash"],
                   status=status, local_synced=False, error=message)
        results.setdefault("job_errors", {})[jid] = rec
        if done_revision(results["jobs"].get(jid)) is None:
            results["jobs"][jid] = rec
        got.add(jid)

    _check_abort()
    account = account or C.account_scope()
    if not account or C.account_scope() != account:
        _fail("無法確認 Colab 登入身分，請重新確認目標 Session", exit_on_error)
    cu_before, _ = C.usage(account=account, deadline=deadline)
    _check_active()
    alive, listing = C.sessions_checked(account=account, deadline=deadline)
    _check_active()
    runtimes = C.listed_runtimes(listing) if alive is not None else None
    if runtimes is None:
        _fail("無法確認開啟中的 Colab Session，請稍後再試", exit_on_error)
    assignments = {ep: name for ep, (name, _) in runtimes.items()}
    active = {name: ep for ep, name in assignments.items() if name != "?"}
    if expected_endpoint is not None and (session is None or active.get(session) != expected_endpoint):
        _fail("所選的 Colab Session 已變更，請重新整理並選擇", exit_on_error)

    allocation_reserved = False
    allocation_attempted = False
    created = False
    owned = False

    got: set[str] = set()
    expected_todo_ids = {j["id"] for j in todo}
    finished = False
    run_lock = None
    cancel_sync_deadline = None
    snapshot = None
    input_shas = {rel: all_input_shas[rel] for rel in sorted({ref for job in todo for ref in job.get("inputs") or []})}

    try:
        _check_active()
        if input_shas:
            snapshot = tempfile.TemporaryDirectory(prefix="cstudio-references-")
            _snapshot_references(base, Path(snapshot.name), input_shas)
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
            listed_hardware = runtimes[endpoint][1]
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
            _pin_versions=pin_versions,
            _pin_models=pin_models,
            _fingerprint_version=2,
        )
        if cli_quant:
            remote_m["_cli_quant"] = cli_quant
        manifest_json = json.dumps(remote_m, ensure_ascii=False, indent=1)

        k_info = C.keep_info(session) or {}
        need_code = created or k_info.get("bundle_hash") != bundle_hash

        remote_run = f"{C.REMOTE_ROOT}/runs/{run_id}"
        archived = {jid for jid, _ in stale}
        request = {
            "schema": 2, "run_id": run_id, "session": session, "account_scope": account,
            "runtime_endpoint": endpoint, "bundle_hash": bundle_hash,
            "gpu": gpu, "model": results["model"],
            "pin_versions": pin_versions, "pin_models": pin_models,
            "jobs": {jid: {"fingerprint": fps[jid], "previous": done_revision(results["jobs"].get(jid)),
                           "archive": jid in archived} for jid in sorted(expected_todo_ids)},
        }
        outputs = RunOutputs(base, run_id, request)
        outputs.directory.mkdir(mode=0o700, parents=True)
        run_lock = outputs.lock()
        run_lock.acquire()
        save_results(outputs.directory / "request.json", request)
        results["bundle_hash"] = bundle_hash
        recover_hint = f"uv run cstudio recover {shlex.quote(str(Path(manifest_path).absolute()))} --run-id {run_id}"
        print(f"📋 批次 {run_id}（成果補抓：{recover_hint}）", flush=True)

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

        _emit({"stage": "syncing_code", "done": False,
               "log": [f"正在同步遠端程式與任務設定至 {session}...", f"成果補抓：{recover_hint}"]})
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
            C.colab("upload", "-s", session, confined_path(Path(snapshot.name), f),
                    f"{remote_run}/uploads/{f}", timeout=600, **target)
        if snapshot is not None:
            snapshot.cleanup()
            snapshot = None

        _check_active()
        if prep["state"] == "prepared":
            launch = C.vm_program("PREP", "launch_run", run_id, bundle_hash, list(REMOTE_CODE_FILES), include_start=True)
            prep = request_preparation(launch)
            if not remote_attempted:
                raise RuntimeError("遠端尚未啟動本批，程式版本或參考圖片未通過驗證")
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

            r_jobs = st.get("jobs", {})
            if not isinstance(r_jobs, dict):
                raise RuntimeError("遠端成果清單格式錯誤")
            if (cancel_pending or st.get("stage") == "cancelled") and cancel_sync_deadline is None:
                cancel_sync_deadline = min(deadline, time.monotonic() + C.RECOVERY_TIMEOUT)
            sync_deadline = cancel_sync_deadline if cancel_sync_deadline is not None else deadline

            def check_sync():
                _check_active(sync_deadline)
                C.remaining_time(C.RECOVERY_TIMEOUT, sync_deadline)

            def download(file, dest):
                C.colab("download", "-s", session, f"{remote_run}/{file}", dest,
                        account=account, endpoint=endpoint, deadline=sync_deadline)

            synced_any = False
            for jid, jinfo in r_jobs.items():
                if jid not in expected_todo_ids or jid in got or not isinstance(jinfo, dict):
                    continue
                if jinfo.get("status") == "done":
                    max_pull_tries = 3 if st.get("done") and cancel_sync_deadline is None else 1
                    for pull_try in range(max_pull_tries):
                        _check_abort()
                        try:
                            rec, _ = outputs.publish(
                                jid, check_sync, timeout=C.remaining_time(WORKSPACE_LOCK_TIMEOUT, sync_deadline),
                            )
                            if rec is None:
                                outputs.pull(jid, jinfo, download, check_sync)
                                rec, _ = outputs.publish(
                                    jid, check_sync, timeout=C.remaining_time(WORKSPACE_LOCK_TIMEOUT, sync_deadline),
                                )
                                if rec is None:
                                    raise RuntimeError("找不到已驗證的暫存成果")
                        except (RunStopped, RunCancelled, C.KeepChanged):
                            raise
                        except SupersededResult as exc:
                            _record_job_error(jid, jinfo, "sync_error", str(exc))
                            synced_any = True
                            print(f"⚠️ {exc}", flush=True)
                            break
                        except (OSError, RuntimeError, ValueError) as exc:
                            _check_active()
                            pull_err = redact(str(exc))
                            if pull_try + 1 < max_pull_tries:
                                print(f"⚠️ 拉回 {jid}.png 失敗，重試中 ({pull_try + 1}/{max_pull_tries})：{pull_err}", flush=True)
                                time.sleep(C.remaining_time(2, sync_deadline))
                            else:
                                print(f"⚠️ 拉回 {jid}.png 失敗：{pull_err}", flush=True)
                            continue

                        got.add(jid)
                        local_synced.add(jid)
                        synced_any = True
                        results["jobs"][jid] = rec
                        print(f"⬇️ 已同步回本地：{out_dir / f'{jid}.png'} ({rec['bytes'] / 1024:.1f} KB)", flush=True)
                        if isinstance(rec.get("refined_prompt"), str) and rec["refined_prompt"]:
                            print(f"   ✨ Prompt Enhancer 擴寫結果：{redact(rec['refined_prompt'])[:160]}...", flush=True)
                        break
                    else:
                        if st.get("done"):
                            _record_job_error(jid, jinfo, "sync_error", f"遠端成果同步失敗：{pull_err}")
                            synced_any = True
                elif jinfo.get("status") == "error":
                    msg = str(jinfo.get("error") or "遠端任務失敗")
                    _record_job_error(jid, jinfo, "error", msg)
                    synced_any = True
                    print(f"❌ 任務 {jid} 失敗：{msg}", flush=True)

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
                if expected_todo_ids - local_synced:
                    errors = results.get("job_errors", {})
                    details = [f"{jid} ({errors[jid]['error']})" if jid in errors else jid
                               for jid in sorted(expected_todo_ids - local_synced)]
                    results.setdefault("error", f"部分任務未成功完成：{', '.join(details)}")
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
                        (stop_after and (finished or created or (owned and not remote_attempted)))
                        or (not finished and ((created and not remote_attempted)
                                              or (owned and time.monotonic() >= deadline)))
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
                batch_error = sys.exception()
                try:
                    cu_after, _ = C.usage(account=account, deadline=cleanup_deadline)
                    cu_used = round(cu_before - cu_after, 2) if (cu_before is not None and cu_after is not None) else None
                    results.update(cu_before=cu_before, cu_after=cu_after, cu_used=cu_used)
                    try:
                        results = workspace.finish(results, prev_results, timeout=max(
                            0, min(WORKSPACE_LOCK_TIMEOUT, cleanup_deadline - time.monotonic()),
                        ))
                    except (OSError, RuntimeError, ValueError) as exc:
                        if batch_error is None:
                            raise
                        print(f"⚠️ 無法寫入批次摘要：{redact(str(exc))}", flush=True)
                    if cu_used is not None:
                        print(f"💳 本批預估消耗 {cu_used} 運算單元（{cu_before} → {cu_after}；Colab 用量可能延遲更新）", flush=True)
                finally:
                    if run_lock is not None:
                        run_lock.release()
                    if snapshot is not None:
                        snapshot.cleanup()
        else:
            try:
                if allocation_reserved:
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
            finally:
                if snapshot is not None:
                    snapshot.cleanup()

    print(f"✅ 批次結束：成功 {len(local_synced)}/{len(m['jobs'])}，耗時 {results['minutes']} 分鐘 → {out_dir}", flush=True)
    if results.get("error") and exit_on_error:
        sys.exit(1)
    return results
