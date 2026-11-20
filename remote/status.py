"""Read run status and coordinate atomic changes to the runtime's active worker."""
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .cancel import CANCEL_PROTOCOL, request_cancel
    from .job_paths import RUN_ID, SHA256, confined_path


def _mkdir(root: Path, relative: str) -> Path:
    path = confined_path(root, relative, directory=True, missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_object(root: Path, name: str) -> dict | None:
    try:
        path = confined_path(root, name)
    except FileNotFoundError:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"遠端紀錄格式錯誤：{name}")
    return data


def _write_object(root: Path, name: str, data: dict):
    path = confined_path(root, name, missing_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".record-") as tmp:
        staged = Path(tmp) / "record.json"
        staged.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        staged.replace(path)


@contextmanager
def _file_lock(root: Path, name: str, *, blocking: bool = True):
    import fcntl

    path = confined_path(root, name, missing_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield fd
    finally:
        # The launched worker inherits the same open file description. Closing this
        # descriptor leaves its lock held until the last descriptor is closed.
        os.close(fd)


def _run_dir(root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("不合法的批次 ID")
    return confined_path(root, f"runs/{run_id}", directory=True, missing_ok=True)


def _legacy_worker_alive(root: Path) -> bool:
    legacy = root / "remote_runner.py"
    if not legacy.is_file():
        return False
    worker = os.fsencode(legacy)
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if argv[1:2] == [worker]:
            return True
    return False


def _read_run_status(job: Path, run_id: str, *, alive: bool, request: dict) -> dict:
    bundle_hash = request.get("bundle_hash")
    if not isinstance(bundle_hash, str) or not SHA256.fullmatch(bundle_hash):
        raise RuntimeError("遠端批次缺少有效的程式版本")
    snapshot = _read_object(job, "status.json")
    started = _read_object(job, "launch.json") is not None
    if snapshot is None:
        snapshot = {
            "run_id": run_id, "bundle_hash": bundle_hash, "cancel_protocol": CANCEL_PROTOCOL,
            "stage": "starting" if started else "waiting_inputs", "jobs": {}, "done": False,
        }
    if snapshot.get("run_id") != run_id or snapshot.get("bundle_hash") != bundle_hash:
        raise RuntimeError("遠端狀態與批次或程式版本不符")
    if started and not alive and not snapshot.get("done"):
        log_path = confined_path(job, "remote_runner.log", missing_ok=True)
        tail = ""
        if log_path.is_file():
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 1500))
                tail = log.read().decode(errors="replace")
        snapshot.update(done=True, stage="crashed", error=snapshot.get("error") or ("the batch process exited\n" + tail))
    snapshot["alive"] = alive
    snapshot["log"] = snapshot.get("log", [])[-10:]
    return snapshot


def _active_run(root: Path) -> str | None:
    active = _read_object(root, "active_run.json")
    if active is None:
        return None
    run_id = active.get("run_id")
    _run_dir(root, run_id)
    return run_id


def _legacy_status(root: Path, *, alive: bool) -> dict:
    st = _read_object(root, "status.json")
    if st is None:
        return {"stage": "not started", "jobs": {}, "done": False, "alive": alive}
    if not alive and not st.get("done"):
        st.update(done=True, stage="crashed", error=st.get("error") or "the legacy batch process exited")
    return dict(st, alive=alive, log=st.get("log", [])[-10:])


def runtime_status(root: Path, run_id: str | None = None) -> dict:
    if run_id is not None and (not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)):
        raise ValueError("不合法的批次 ID")
    if not root.exists() and not root.is_symlink():
        return {"run_id": run_id, "stage": "not started", "jobs": {}, "done": False, "alive": False}
    with _file_lock(root, "control.lock"):
        active = _active_run(root)
        with _file_lock(root, "worker.lock", blocking=False) as fd:
            busy = fd is None
        if busy and active is None:
            raise RuntimeError("無法確認執行中批次的識別")
        legacy_alive = _legacy_worker_alive(root)
        if run_id is None and legacy_alive:
            return _legacy_status(root, alive=True)
        target = run_id or active
        if target is not None:
            job = _run_dir(root, target)
            request = _read_object(job, "request.json")
            if request is not None:
                st = _read_run_status(job, target, alive=busy and active == target, request=request)
                return dict(st, active_run_id=active if busy else None, legacy_alive=legacy_alive)
            if target == active:
                raise RuntimeError("找不到執行中批次的遠端紀錄")
        st = _legacy_status(root, alive=legacy_alive)
        if run_id is not None and st.get("run_id") != run_id:
            return {"run_id": run_id, "stage": "not started", "jobs": {}, "done": False, "alive": False,
                    "active_run_id": active if busy else None, "legacy_alive": legacy_alive}
        return st


def _mark_cancel(job: Path, run_id: str):
    _mkdir(job, "cancel_requests")
    marker = confined_path(job, f"cancel_requests/{run_id}", missing_ok=True)
    marker.touch()


def _cancel_unstarted(job: Path, run_id: str, request: dict):
    _write_object(job, "status.json", {
        "run_id": run_id, "bundle_hash": request["bundle_hash"], "cancel_protocol": CANCEL_PROTOCOL,
        "stage": "cancelled", "done": True, "jobs": {},
    })


def request_run_cancel(root: Path, run_id: str | None, allow_start: bool = False) -> dict:
    if run_id is not None and (not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)):
        raise ValueError("不合法的批次 ID")
    result = {"run_id": run_id} if run_id is not None else {}
    if not root.exists() and not root.is_symlink():
        if not allow_start or run_id is None:
            return dict(result, state="idle")
        _mkdir(root.parent, root.name)
    with _file_lock(root, "control.lock"):
        active = _active_run(root)
        legacy = _legacy_status(root, alive=_legacy_worker_alive(root))
        if run_id is None and legacy.get("alive") and legacy.get("run_id"):
            return request_cancel(root, legacy["run_id"])
        target = run_id or active
        if target is not None:
            job = _run_dir(root, target)
            request = _read_object(job, "request.json")
            if request is not None:
                with _file_lock(root, "worker.lock", blocking=False) as fd:
                    alive = fd is None and target == active
                st = _read_run_status(job, target, alive=alive, request=request)
                if st.get("done"):
                    state = {"cancelled": "confirmed", "cancel_failed": "failed"}.get(st["stage"], "finished")
                elif not alive and _read_object(job, "launch.json") is None:
                    _cancel_unstarted(job, target, request)
                    state = "confirmed"
                else:
                    _mark_cancel(job, target)
                    state = "pending"
                return {"run_id": target, "state": state}
        if run_id is None or legacy.get("run_id") == run_id:
            return request_cancel(root, run_id)
        if allow_start:
            _mark_cancel(root, run_id)
            return dict(result, state="pending")
        return dict(result, state="idle")
