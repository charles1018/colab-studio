"""Record a cancellation request for a specific Studio run on the VM."""
import json
import re
from pathlib import Path

CANCEL_PROTOCOL = 1


def cancel_marker(job: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise ValueError("Invalid Studio run ID")
    return job / "cancel_requests" / run_id


def request_cancel(job: Path, run_id: str | None, allow_start: bool = False) -> dict:
    try:
        snapshot = json.loads((job / "status.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        snapshot = {}

    run_id = run_id or snapshot.get("run_id")
    if not run_id:
        return {"state": "idle"}
    marker = cancel_marker(job, run_id)
    result = {"run_id": run_id}

    if snapshot.get("run_id") == run_id:
        if snapshot.get("done"):
            stage = snapshot.get("stage")
            if stage not in ("cancelled", "cancel_failed"):
                return dict(result, state="finished")
        if snapshot.get("cancel_protocol") != CANCEL_PROTOCOL:
            return dict(result, state="unsupported")
        if snapshot.get("done"):
            if stage == "cancelled":
                return dict(result, state="confirmed")
            return dict(result, state="failed")
    elif not allow_start:
        return dict(result, state="idle")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return dict(result, state="pending")
