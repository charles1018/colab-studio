"""Verify and publish run outputs with local recovery journals."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout as LockTimeout

from remote.job_paths import JOB_ID, RUN_ID, SHA256, confined_path, file_sha256, json_digest

WORKSPACE_LOCK_TIMEOUT = 5
RUN_SUMMARY_FIELDS = {
    "session", "gpu", "run_id", "model", "bundle_hash", "minutes", "error",
    "shutdown_state", "shutdown_pending", "cu_before", "cu_after", "cu_used",
}


def read_object(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"成果紀錄必須是 JSON object（{path.name}）")
    return data


def save_results(path: Path, data: dict):
    confined_path(path.parent, path.name, missing_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".record-", encoding="utf-8", delete=False,
        ) as stream:
            staged = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def is_sha256(value) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def done_revision(info) -> str | None:
    if not isinstance(info, dict) or info.get("status") != "done":
        return None
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode("utf-8")).hexdigest()


def verify_receipt(jid: str, info: dict):
    if info.get("status") != "done" or info.get("file") != f"out/{jid}.png":
        raise ValueError("遠端成果路徑或狀態與當批任務不符")
    if type(info.get("bytes")) is not int or info["bytes"] <= 0:
        raise ValueError("遠端成果缺少有效的檔案大小")
    if not is_sha256(info.get("sha256")):
        raise ValueError("遠端成果缺少有效的 SHA-256")


def matches_image(path: Path, info: dict) -> bool:
    return path.is_file() and path.stat().st_size == info["bytes"] and file_sha256(path) == info["sha256"]


def execution_digest(fingerprint: str, execution, *, pin_versions: bool, pin_models: bool) -> str:
    if (
        not isinstance(execution, dict)
        or execution.get("pin_versions") is not pin_versions or execution.get("pin_models") is not pin_models
        or not isinstance(execution.get("environment"), dict)
        or not isinstance(execution.get("gpu"), dict) or not execution["gpu"].get("name")
        or not isinstance(execution.get("comfy"), dict)
        or type(execution["comfy"].get("verified")) is not bool
        or not isinstance(execution["comfy"].get("args"), list) or not execution["comfy"]["args"]
        or any(not isinstance(arg, str) for arg in execution["comfy"]["args"])
        or not isinstance(execution["comfy"].get("versions"), dict)
        or not isinstance(execution.get("models"), list) or not execution["models"]
        or not isinstance(execution.get("fitted_models"), list)
        or (pin_models and execution["fitted_models"])
    ):
        raise ValueError("遠端成果缺少與批次相符的執行環境紀錄")
    environment = execution["environment"]
    if (
        not isinstance(environment.get("python"), list) or len(environment["python"]) != 2
        or not isinstance(environment.get("heads"), dict) or not environment["heads"]
        or not isinstance(environment.get("packages"), dict)
    ):
        raise ValueError("遠端成果缺少有效的 Python、套件或 Git 版本紀錄")
    for model in execution["models"]:
        if (
            not isinstance(model, dict)
            or not all(isinstance(model.get(key), str) and model[key] for key in ("repo", "file", "path"))
            or type(model.get("bytes")) is not int or model["bytes"] <= 0
            or (pin_models and (
                not is_sha256(model.get("sha256"))
                or not isinstance(model.get("revision"), str) or not model["revision"]
            ))
        ):
            raise ValueError("遠端成果缺少有效的模型檔案紀錄")
    return json_digest({"schema": 2, "request": fingerprint, "execution": execution})


def _copy_image(source: Path, dest: Path):
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".image-") as directory:
        staged = Path(directory) / "image.png"
        with source.open("rb") as original, staged.open("wb") as copy:
            shutil.copyfileobj(original, copy)
            copy.flush()
            os.fsync(copy.fileno())
        staged.replace(dest)


class SupersededResult(RuntimeError):
    """Another run published a result for this job after this run began."""


class WorkspaceResults:
    def __init__(self, base: Path):
        self.base = base

    def _path(self, name: str) -> Path:
        return confined_path(self.base, f"out/{name}", missing_ok=True)

    @property
    def path(self) -> Path:
        return self._path("results.json")

    def backup_path(self, run_id: str, jid: str, archive: bool) -> Path:
        if archive:
            return self._path(f"_stale/{jid}_{run_id}.png")
        return self._path(f".runs/{run_id}/{jid}.previous.png")

    @contextmanager
    def transaction(self, timeout: float = WORKSPACE_LOCK_TIMEOUT):
        lock = FileLock(self._path(".results.lock"), timeout=timeout)
        try:
            lock.acquire()
        except LockTimeout:
            raise RuntimeError("工作目錄正在發布成果，請稍後重試") from None
        try:
            current = read_object(self.path)
            if not isinstance(current.setdefault("jobs", {}), dict):
                raise ValueError("results.json 的成果清單格式錯誤")
            self.restore_interrupted_image(current)
            yield current
        finally:
            lock.release()

    def mark_image_replace(self, jid: str, info: dict, *, had_image: bool, archive: bool):
        save_results(self._path(".publish.json"), {
            "jid": jid, "run_id": info["run_id"], "revision": done_revision(info),
            "bytes": info["bytes"], "sha256": info["sha256"], "had_image": had_image, "archive": archive,
        })

    def restore_interrupted_image(self, current: dict):
        """Undo an uncommitted replacement before another transaction sees the image."""
        path = self._path(".publish.json")
        pending = read_object(path)
        if not pending and not path.exists():
            return
        jid, run_id = pending.get("jid"), pending.get("run_id")
        if (
            not isinstance(jid, str) or not JOB_ID.fullmatch(jid)
            or not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)
            or not is_sha256(pending.get("revision")) or not is_sha256(pending.get("sha256"))
            or type(pending.get("bytes")) is not int or pending["bytes"] <= 0
            or type(pending.get("had_image")) is not bool or type(pending.get("archive")) is not bool
        ):
            raise ValueError("工作目錄的成果發布紀錄格式錯誤")
        if done_revision(current.get("jobs", {}).get(jid)) != pending["revision"]:
            final = self._path(f"{jid}.png")
            if matches_image(final, pending):
                if pending["had_image"]:
                    _copy_image(self.backup_path(run_id, jid, pending["archive"]), final)
                else:
                    final.unlink()
        path.unlink()

    def finish(self, summary: dict, previous: dict, *, timeout: float) -> dict:
        """Merge this batch's summary and failures into the latest workspace state."""
        with self.transaction(timeout) as current:
            jobs = current["jobs"]
            stored_errors = current.get("job_errors")
            errors = dict(stored_errors) if isinstance(stored_errors, dict) else {}
            old_errors = previous.get("job_errors")
            if not isinstance(old_errors, dict):
                old_errors = {}
            for jid, error in summary.get("job_errors", {}).items():
                if jobs.get(jid) != previous["jobs"].get(jid) or errors.get(jid) != old_errors.get(jid):
                    continue
                errors[jid] = error
                if done_revision(jobs.get(jid)) is None:
                    jobs[jid] = error
            if errors:
                current["job_errors"] = errors
            for field in RUN_SUMMARY_FIELDS:
                current.pop(field, None)
            current.update({key: value for key, value in summary.items() if key in RUN_SUMMARY_FIELDS})
            save_results(self.path, current)
            return current


class RunOutputs:
    def __init__(self, base: Path, run_id: str, request: dict):
        self.base = base
        self.run_id = run_id
        self.request = request
        self.directory = confined_path(base, f"out/.runs/{run_id}", directory=True, missing_ok=True)
        self.workspace = WorkspaceResults(base)

    def _path(self, name: str) -> Path:
        return confined_path(self.directory, name, missing_ok=True)

    def _destination(self, jid: str) -> Path:
        return confined_path(self.base, f"out/{jid}.png", missing_ok=True)

    def lock(self) -> FileLock:
        return FileLock(self._path("publish.lock"), timeout=0)

    def _verified_record(self, jid: str, info: dict) -> dict:
        verify_receipt(jid, info)
        fingerprint = self.request["jobs"][jid]["fingerprint"]
        record = dict(info, run_id=self.run_id, bundle_hash=self.request["bundle_hash"],
                      fingerprint=fingerprint, local_synced=True)
        if self.request["schema"] == 2:
            record["fingerprint_version"] = 2
            record["execution_fingerprint"] = execution_digest(
                fingerprint, info.get("execution"), pin_versions=self.request["pin_versions"],
                pin_models=self.request["pin_models"],
            )
        return record

    def _published(self, jid: str, info) -> bool:
        if not isinstance(info, dict) or info.get("run_id") != self.run_id:
            return False
        try:
            expected = self._verified_record(jid, info)
        except ValueError:
            return False
        return expected == info and matches_image(self._destination(jid), info)

    def _check_previous(self, jid: str, info):
        if isinstance(info, dict) and info.get("run_id") == self.run_id:
            return
        if done_revision(info) != self.request["jobs"][jid]["previous"]:
            raise SupersededResult(f"{jid} 的本機成果紀錄已由其他批次更新")

    def pull(self, jid: str, info: dict, download, check_active):
        record = self._verified_record(jid, info)
        with tempfile.TemporaryDirectory(dir=self.directory, prefix=".download-") as directory:
            check_active()
            download(record["file"], Path(directory) / "image.png")
            check_active()
            pulled = confined_path(Path(directory), "image.png")
            if pulled.stat().st_size != record["bytes"]:
                raise RuntimeError("下載檔案大小不符")
            if not matches_image(pulled, record):
                raise RuntimeError("下載檔案 SHA-256 不符")
            _copy_image(pulled, self._path(f"{jid}.png"))
        save_results(self._path(f".result-{jid}.json"), record)

    def publish(self, jid: str, check_active, *, timeout: float = WORKSPACE_LOCK_TIMEOUT) -> tuple[dict | None, bool]:
        """Return the verified local record and whether this call committed it."""
        with self.workspace.transaction(timeout) as current:
            check_active()
            info = current["jobs"].get(jid)
            self._check_previous(jid, info)
            if self._published(jid, info):
                return info, False
            return self._publish(jid, current, check_active)

    def _publish(self, jid: str, current: dict, check_active) -> tuple[dict | None, bool]:
        receipt = self._path(f".result-{jid}.json")
        if not receipt.is_file():
            return None, False
        info = read_object(receipt)
        if info != self._verified_record(jid, info):
            raise ValueError("暫存成果的批次或程式版本不符")
        staged = self._path(f"{jid}.png")
        final = self._destination(jid)
        already_replaced = matches_image(final, info)
        if not already_replaced and not matches_image(staged, info):
            raise RuntimeError("暫存成果大小或 SHA-256 不符")
        archive = self.request["jobs"][jid]["archive"]
        backup = self.workspace.backup_path(self.run_id, jid, archive)
        had_image = final.is_file()
        had_backup = backup.is_file()
        if had_image and not had_backup and (archive or not already_replaced):
            backup.parent.mkdir(exist_ok=True)
            _copy_image(final, backup)
        updated = dict(current, jobs={**current["jobs"], jid: info})
        if current.keys() == {"jobs"} and not current["jobs"]:
            updated.update(session=self.request["session"], run_id=self.run_id, bundle_hash=self.request["bundle_hash"])
            updated.update({key: self.request[key] for key in ("gpu", "model") if key in self.request})
        errors = current.get("job_errors")
        if isinstance(errors, dict) and jid in errors:
            updated["job_errors"] = {key: value for key, value in errors.items() if key != jid}
        try:
            # A pre-upgrade journal may already have installed its image and kept the backup.
            if not already_replaced or had_backup:
                check_active()
                self.workspace.mark_image_replace(jid, info, had_image=had_image, archive=archive)
            if not already_replaced:
                _copy_image(staged, final)
            check_active()
            save_results(self.workspace.path, updated)
        except BaseException:
            self.workspace.restore_interrupted_image(read_object(self.workspace.path))
            raise
        self.workspace.restore_interrupted_image(updated)
        # A committed pair takes precedence over leftover staging files.
        try:
            receipt.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
            if not archive:
                backup.unlink(missing_ok=True)
        except OSError:
            pass
        return info, True
