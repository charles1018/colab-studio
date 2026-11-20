"""Verify and publish run outputs with a per-job local recovery journal."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock

from remote.job_paths import SHA256, confined_path


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
    if not path.is_file() or path.stat().st_size != info["bytes"]:
        return False
    with path.open("rb") as image:
        return hashlib.file_digest(image, "sha256").hexdigest() == info["sha256"]


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


class RunOutputs:
    def __init__(self, base: Path, run_id: str, request: dict):
        self.base = base
        self.run_id = run_id
        self.request = request
        self.directory = confined_path(base, f"out/.runs/{run_id}", directory=True, missing_ok=True)
        self.results_path = confined_path(base, "out/results.json", missing_ok=True)

    def _path(self, name: str) -> Path:
        return confined_path(self.directory, name, missing_ok=True)

    def _destination(self, jid: str) -> Path:
        return confined_path(self.base, f"out/{jid}.png", missing_ok=True)

    def lock(self) -> FileLock:
        return FileLock(self._path("publish.lock"), timeout=0)

    def _verified_record(self, jid: str, info: dict) -> dict:
        verify_receipt(jid, info)
        return dict(info, run_id=self.run_id, bundle_hash=self.request["bundle_hash"],
                    fingerprint=self.request["jobs"][jid]["fingerprint"], local_synced=True)

    def published(self, jid: str, results: dict) -> bool:
        info = results["jobs"].get(jid)
        if not isinstance(info, dict) or info.get("run_id") != self.run_id:
            return False
        try:
            expected = self._verified_record(jid, info)
        except ValueError:
            return False
        return expected == info and matches_image(self._destination(jid), info)

    def check_previous(self, jid: str, results: dict):
        info = results["jobs"].get(jid)
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

    def publish(self, jid: str, results: dict, check_active) -> bool:
        """Replay a completed download, even if the source runtime has gone."""
        receipt = self._path(f".result-{jid}.json")
        if not receipt.is_file():
            return False
        info = read_object(receipt)
        if info != self._verified_record(jid, info):
            raise ValueError("暫存成果的批次或程式版本不符")
        self.check_previous(jid, results)
        staged = self._path(f"{jid}.png")
        final = self._destination(jid)
        already_replaced = matches_image(final, info)
        if not already_replaced and not matches_image(staged, info):
            raise RuntimeError("暫存成果大小或 SHA-256 不符")
        archive = self.request["jobs"][jid]["archive"]
        backup = self._path(f"{jid}.previous.png")
        if archive:
            backup = confined_path(self.base, f"out/_stale/{jid}_{self.run_id}.png", missing_ok=True)
        check_active()
        installed = False
        if final.is_file() and not backup.is_file() and (archive or not already_replaced):
            backup.parent.mkdir(exist_ok=True)
            _copy_image(final, backup)
        if not already_replaced:
            check_active()
            _copy_image(staged, final)
            installed = True
        current = dict(results, jobs={**results["jobs"], jid: info})
        errors = results.get("job_errors")
        if results.get("run_id") == self.run_id and isinstance(errors, dict) and jid in errors:
            current["job_errors"] = {key: value for key, value in errors.items() if key != jid}
        try:
            check_active()
            save_results(self.results_path, current)
            results.update(current)
        except BaseException:
            # A process kill can bypass rollback. The journal still has the
            # verified bytes and the previous image for the next invocation.
            stored = read_object(self.results_path).get("jobs", {})
            if isinstance(stored, dict) and stored.get(jid) == info:
                results.update(current)
            elif backup.is_file():
                _copy_image(backup, final)
            elif installed:
                final.unlink()
            raise
        # A committed pair takes precedence over leftover staging files.
        try:
            receipt.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
            if not archive:
                backup.unlink(missing_ok=True)
        except OSError:
            pass
        return True
