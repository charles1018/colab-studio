"""Reconcile Colab assignments and local leases for the workbench."""
from __future__ import annotations

import hmac
import json
import math
from dataclasses import dataclass, field
from typing import Any

from . import colab_client as C
from .redact import redact

CACHE_SECONDS = 30
STALE_SECONDS = 90


@dataclass
class CloudStatus:
    # Keep the last successful list available for context through a temporary failure.
    runtimes: dict[str, tuple[str, str]] = field(default_factory=dict)
    complete: bool = False
    checked_at: float | None = None
    checked_mono: float | None = None
    confirmed_at: float | None = None
    cu: float | None = None
    cu_raw: str = "尚未查詢"
    failed: bool = False
    cu_failed: bool = False

    def fresh(self, now: float, limit: float) -> bool:
        return self.checked_mono is not None and now - self.checked_mono < limit


def session_id(key: bytes, account: str, name: str, endpoint: str) -> str:
    return hmac.digest(key, json.dumps([account, name, endpoint]).encode(), "sha256").hex()


def session_rows(
    cache: CloudStatus, account: str, key: bytes, now: float, complete: bool,
) -> list[dict[str, Any]]:
    records: dict[str, dict | None] = {}
    for path in sorted(C.KEEP.glob("*.json")):
        try:
            C.sanitize_session_name(path.stem)
        except ValueError:
            continue
        records[path.stem] = C.keep_info(path.stem)

    rows = []
    for endpoint, (name, hardware) in cache.runtimes.items():
        if name == "?":
            continue
        try:
            C.sanitize_session_name(name)
        except ValueError:
            continue
        record = records.get(name)
        same = bool(record and record.get("account_scope", account) == account
                    and record.get("runtime_endpoint") == endpoint)
        lease = record if same and (complete or record.get("account_scope") == account) else None
        blocked = name in records and (
            record is None or record.get("account_scope", account) != account
            or record.get("runtime_endpoint") not in (None, endpoint) or record.get("created_by") == "pending"
        )
        rows.append(_row(key, account, name, endpoint, hardware, "present" if complete else "unknown",
                         lease, now, blocked=blocked))

    for name, record in records.items():
        # A legacy keep only belongs in the view after a positive alias/endpoint match.
        if not record or record.get("account_scope") != account:
            continue
        endpoint = record.get("runtime_endpoint")
        if endpoint and cache.runtimes.get(endpoint, (None,))[0] == name:
            continue
        pending = record.get("created_by") == "pending"
        gone = complete and endpoint and endpoint not in cache.runtimes
        rows.append(_row(key, account, name, endpoint, None, "gone" if gone else "unknown",
                         record, now, blocked=True, pending=pending))
    return rows


def _row(key, account, name, endpoint, hardware, presence, lease, now, *, blocked, pending=False):
    lease = lease or {}
    owned = lease.get("created_by") == "owned" and bool(endpoint)
    release_pending = bool(lease.get("release_pending"))
    expires = lease.get("expires")
    busy = bool(lease) and not pending and (bool(lease.get("active_run_id")) or expires is None)
    remaining = None
    if not busy and not release_pending and type(expires) in (int, float) and math.isfinite(expires):
        remaining = max(0, round(expires - now))
    return {
        "id": session_id(key, account, name, endpoint) if endpoint else None,
        "name": name,
        "hardware": redact(hardware) if hardware else None,
        "presence": presence,
        "owned": owned,
        "local_state": ("release_pending" if release_pending else "allocating" if pending else
                        "busy" if busy else "idle" if remaining is not None else None),
        "remaining_sec": remaining,
        "blocked": blocked,
        "reusable": (presence == "present" and not blocked and not release_pending
                     and not (owned and lease.get("active_run_id"))),
    }
