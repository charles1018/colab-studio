"""Prepare a run's immutable request and start one worker per Colab VM."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

if __package__:
    from .job_paths import SHA256, bundle_digest, confined_path
    from .status import (
        _active_run, _cancel_unstarted, _file_lock, _legacy_status, _legacy_worker_alive,
        _mkdir, _read_object, _read_run_status, _run_dir, _write_object,
    )


def _receipt(run_id: str, state: str, bundle_hash: str | None = None, **extra) -> dict:
    return {"run_id": run_id, "state": state, "bundle_hash": bundle_hash, **extra}


def _installed_bundle(root: Path, bundle_hash: str, names: list[str]) -> Path | None:
    bundle = confined_path(root, f"bundles/{bundle_hash}", directory=True, missing_ok=True)
    try:
        contents = {name: confined_path(bundle, name).read_text(encoding="utf-8") for name in names}
    except (FileNotFoundError, UnicodeError):
        return None
    return bundle if bundle_digest(contents) == bundle_hash else None


def _publish_bundle(root: Path, bundle_hash: str, names: list[str], files: dict[str, str]) -> Path:
    if sorted(files) != sorted(names) or not all(isinstance(source, str) for source in files.values()):
        raise RuntimeError("遠端程式清單不符")
    if bundle_digest(files) != bundle_hash:
        raise RuntimeError("遠端程式摘要不符")
    parent = _mkdir(root, "bundles")
    bundle = confined_path(parent, bundle_hash, directory=True, missing_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".install-") as tmp:
        staged = Path(tmp) / "bundle"
        staged.mkdir()
        for name, source in files.items():
            path = confined_path(staged, name, missing_ok=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        if bundle.exists():
            bundle.replace(parent / f".invalid-{uuid.uuid4().hex}")
        staged.replace(bundle)
    return bundle


def _file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _missing_inputs(job: Path, inputs: dict[str, str], folder: str) -> list[str]:
    missing = []
    for name, expected in inputs.items():
        path = confined_path(job, f"{folder}/{name}", missing_ok=True)
        if not path.is_file() or _file_sha(path) != expected:
            missing.append(name)
    return missing


def _seal_inputs(job: Path, inputs: dict[str, str]) -> list[str]:
    sealed = confined_path(job, "inputs", directory=True, missing_ok=True)
    if sealed.exists():
        if _missing_inputs(job, inputs, "inputs"):
            raise RuntimeError("已驗證的參考圖片已變更")
        return []
    with tempfile.TemporaryDirectory(dir=job, prefix=".inputs-") as tmp:
        staged = Path(tmp) / "inputs"
        staged.mkdir()
        missing = []
        for name, expected in inputs.items():
            source = confined_path(job, f"uploads/{name}", missing_ok=True)
            if not source.is_file():
                missing.append(name)
                continue
            dest = confined_path(staged, name, missing_ok=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            if _file_sha(dest) != expected:
                missing.append(name)
        if not missing:
            staged.replace(sealed)
        return missing


def _existing_launch(job: Path, run_id: str, request: dict, *, alive: bool) -> dict | None:
    if _read_object(job, "launch.json") is None and _read_object(job, "status.json") is None:
        return None
    st = _read_run_status(job, run_id, alive=alive, request=request)
    return _receipt(run_id, "finished" if st["done"] else "running", st["bundle_hash"], missing_inputs=[])


def _worker_conflict(root: Path, active: str | None, busy: bool) -> str | None:
    if busy or _legacy_worker_alive(root):
        return "busy"
    if active is not None:
        job = _run_dir(root, active)
        request = _read_object(job, "request.json")
        if request is None:
            raise RuntimeError("找不到執行中批次的遠端紀錄")
        st = _read_run_status(job, active, alive=False, request=request)
    else:
        st = _legacy_status(root, alive=False)
    return "cancel_failed" if st.get("stage") == "cancel_failed" else None


def _launch_worker(root: Path, job: Path, run_id: str, request: dict, bundle: Path, lease: int) -> dict:
    manifest = confined_path(job, "manifest.json")
    if _file_sha(manifest) != request["manifest_sha256"]:
        raise RuntimeError("遠端 manifest 與已登記的批次不符")
    missing = _seal_inputs(job, request["input_shas"])
    bundle_hash = request["bundle_hash"]
    if missing:
        return _receipt(run_id, "prepared", bundle_hash, missing_inputs=missing)
    worker = confined_path(bundle, "remote_runner.py")
    log_path = confined_path(job, "remote_runner.log", missing_ok=True)
    _mkdir(job, "out")
    env = dict(os.environ, CSTUDIO_DIR=str(job), CSTUDIO_RUNTIME_DIR=str(root),
               CSTUDIO_RUN_ID=run_id, CSTUDIO_BUNDLE_HASH=bundle_hash)
    with log_path.open("wb") as log:
        _write_object(root, "active_run.json", {"run_id": run_id})
        intent = confined_path(job, "launch.json", missing_ok=True)
        _write_object(job, "launch.json", {"run_id": run_id})
        try:
            # Older clients recognize the historical path in process arguments.
            process = subprocess.Popen(
                [sys.executable, str(worker), str(root / "remote_runner.py")],
                cwd=str(job), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(lease,), start_new_session=True,
            )
        except OSError:
            intent.unlink()
            raise
    return _receipt(run_id, "running", bundle_hash, missing_inputs=[], pid=process.pid)


def prepare_run(
    root: Path, bundle_hash: str, names: list[str], files: dict[str, str] | None,
    manifest_json: str, input_shas: dict[str, str],
) -> dict:
    manifest = json.loads(manifest_json)
    if not isinstance(manifest, dict) or not isinstance(bundle_hash, str) or not SHA256.fullmatch(bundle_hash):
        raise ValueError("無效的遠端 manifest 或程式版本")
    if not isinstance(input_shas, dict) or any(
        not isinstance(sha, str) or not SHA256.fullmatch(sha) for sha in input_shas.values()
    ):
        raise ValueError("參考圖片摘要格式錯誤")
    run_id = manifest.get("_run_id")
    _mkdir(root.parent, root.name)
    job = _run_dir(root, run_id)
    request = {"bundle_hash": bundle_hash, "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
               "input_shas": input_shas}
    with _file_lock(root, "control.lock"), _file_lock(root, "worker.lock", blocking=False) as lease:
        active = _active_run(root)
        previous = _read_object(job, "request.json")
        if previous is not None:
            if previous != request:
                raise RuntimeError("相同批次 ID 已用於不同的遠端請求")
            replay = _existing_launch(job, run_id, previous, alive=lease is None and active == run_id)
            if replay is not None:
                return replay
        conflict = _worker_conflict(root, active, lease is None)
        if conflict is not None:
            return _receipt(run_id, "busy", reason=conflict)
        bundle = _installed_bundle(root, bundle_hash, names)
        if bundle is None:
            if files is None:
                return _receipt(run_id, "bundle_required")
            bundle = _publish_bundle(root, bundle_hash, names, files)
        if previous is None:
            parent = _mkdir(root, "runs")
            with tempfile.TemporaryDirectory(dir=parent, prefix=".prepare-") as tmp:
                staged = Path(tmp) / "run"
                staged.mkdir()
                _mkdir(staged, "uploads")
                for name in input_shas:
                    upload = confined_path(staged, f"uploads/{name}", missing_ok=True)
                    upload.parent.mkdir(parents=True, exist_ok=True)
                (staged / "manifest.json").write_bytes(manifest_json.encode("utf-8"))
                _write_object(staged, "request.json", request)
                staged.replace(job)
        marker = confined_path(root, f"cancel_requests/{run_id}", missing_ok=True)
        if marker.is_file():
            _cancel_unstarted(job, run_id, request)
            return _receipt(run_id, "finished", bundle_hash, missing_inputs=[])
        if input_shas:
            return _receipt(run_id, "prepared", bundle_hash, missing_inputs=_missing_inputs(job, input_shas, "uploads"))
        return _launch_worker(root, job, run_id, request, bundle, lease)


def launch_run(root: Path, run_id: str, bundle_hash: str, names: list[str]) -> dict:
    with _file_lock(root, "control.lock"), _file_lock(root, "worker.lock", blocking=False) as lease:
        job = _run_dir(root, run_id)
        request = _read_object(job, "request.json")
        if request is None or not isinstance(bundle_hash, str) or not SHA256.fullmatch(bundle_hash) or request.get("bundle_hash") != bundle_hash:
            raise RuntimeError("找不到與批次及版本相符的遠端準備紀錄")
        active = _active_run(root)
        replay = _existing_launch(job, run_id, request, alive=lease is None and active == run_id)
        if replay is not None:
            return replay
        conflict = _worker_conflict(root, active, lease is None)
        if conflict is not None:
            return _receipt(run_id, "busy", reason=conflict)
        bundle = _installed_bundle(root, bundle_hash, names)
        if bundle is None:
            return _receipt(run_id, "bundle_required")
        return _launch_worker(root, job, run_id, request, bundle, lease)
