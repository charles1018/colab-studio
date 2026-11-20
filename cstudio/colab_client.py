# -*- coding: utf-8 -*-
"""Google Colab CLI wrapper, session manager, and background idle-shutdown watchdog."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Callable

from filelock import FileLock, Timeout as LockTimeout

from .colab_login import account_scope
from .redact import redact, redact_diagnostics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = ROOT / "remote"
REMOTE_ROOT = "/content/cstudio"
GPUS = ("L4", "G4", "A100", "H100", "T4")
PREFIX = "cstudio-"
ASSIGN_TIMEOUT = 900
ALLOCATION_REFUSED = "[colab] Allocation refused (precondition failed)."
RECOVERY_TIMEOUT = 30
WATCHDOG_TICK = 30
WATCHDOG_START_TIMEOUT = 10
IDLE_ACTIVE_GRACE = 10 * 60
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
KEEP = Path.home() / ".config" / "colab-studio" / "keep"
_ANY_LEASE = object()


def colab_exe() -> str:
    venv_bin = Path(sys.executable).resolve().parent
    candidates = (
        venv_bin / "colab.exe",
        venv_bin / "colab",
        ROOT / ".venv" / "bin" / "colab",
        ROOT / ".venv" / "Scripts" / "colab.exe",
        shutil.which("colab"),
        Path.home() / ".local" / "bin" / "colab.exe",
        Path.home() / ".local" / "bin" / "colab",
    )
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    sys.exit("找不到 colab 指令：請先在此 repo 執行 `uv sync && uv run cstudio-patch`")


class ColabTimeout(RuntimeError):
    """The command timed out before its outcome could be observed."""


class ColabDeadline(ColabTimeout):
    """The time budget for the operation is exhausted."""


class RuntimeUnavailable(RuntimeError):
    """The saved runtime or its remote run is confirmed absent."""


def remaining_time(limit: float, deadline: float | None) -> float:
    if deadline is None:
        return limit
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ColabDeadline("已到達 Colab 批次時間上限")
    return min(limit, remaining)


def colab(
    *args, timeout: int = 600, check: bool = True,
    account: str | None = None, endpoint: str | None = None, deadline: float | None = None,
) -> str:
    operation = str(args[0]) if args else "指令"
    env = ENV
    command = [colab_exe()] if account is None else [sys.executable, "-m", "cstudio._colab"]
    if account is not None:
        env = dict(ENV, CSTUDIO_COLAB_ACCOUNT_SCOPE=account)
        env.pop("CSTUDIO_COLAB_ENDPOINT", None)
        if endpoint is not None:
            env["CSTUDIO_COLAB_ENDPOINT"] = endpoint
    timeout_left = remaining_time(timeout, deadline)
    try:
        p = subprocess.run(
            [*command, *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_left,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        if deadline is not None and (timeout_left < timeout or time.monotonic() >= deadline):
            raise ColabDeadline("已到達 Colab 批次時間上限，遠端結果待確認") from None
        if check:
            raise ColabTimeout(f"colab {operation} 超過 {timeout} 秒沒有回應") from None
        return ""
    except OSError:
        raise RuntimeError(f"無法啟動 colab {operation}") from None
    stdout, stderr = p.stdout or "", p.stderr or ""
    if p.returncode and check:
        lines = stdout.splitlines() + stderr.splitlines()
        # Preserve the pinned client's explicit refusal receipt for allocation retry logic.
        refusal = operation == "new" and any(line.startswith(ALLOCATION_REFUSED) for line in lines)
        hint = f"\n{ALLOCATION_REFUSED}" if refusal else ""
        if not refusal and operation == "new" and any(
            line.startswith("[colab] Backend rejected accelerator ") for line in lines
        ):
            hint = "\nColab 拒絕所選的 GPU 規格，請檢查帳號配額"
        raise RuntimeError(f"colab {operation} 失敗（{p.returncode}）{hint}")
    if operation == "exec":
        # A remote status/exec receipt can exist even if the local CLI later exits nonzero.
        return "\n".join(line for line in (stdout, redact(stderr)) if line)
    return "" if p.returncode else redact("\n".join(line for line in (stdout, stderr) if line))


def new_session_name() -> str:
    return f"{PREFIX}{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def new_runtime(
    session: str,
    gpu: str,
    retries: int = 5,
    wait: int = 120,
    check_abort: Callable[[], None] | None = None,
    deadline: float | None = None,
    *,
    account: str | None = None,
) -> str:
    account = account or account_scope()
    if not account:
        raise RuntimeError("無法確認 Colab 登入身分")
    for attempt in range(retries + 1):
        if check_abort:
            check_abort()
        try:
            timeout = remaining_time(ASSIGN_TIMEOUT, deadline)
            out = colab("new", "-s", session, "--gpu", gpu, timeout=timeout, account=account, deadline=deadline)
        except RuntimeError as e:
            # Only the upstream explicit 412 rejection proves this request did not allocate.
            refused = any(line.startswith(ALLOCATION_REFUSED) for line in str(e).splitlines())
            if isinstance(e, ColabTimeout) or attempt == retries or not refused or runtime_endpoint(session, deadline=deadline):
                raise
            print(
                f"⏳ Colab 拒絕配置 {gpu}，{wait:g} 秒後再試（第 {attempt + 1}/{retries} 次重試）",
                flush=True,
            )
            if check_abort:
                check_abort()
            time.sleep(remaining_time(wait, deadline))
        else:
            receipts = re.findall(r"^CSTUDIO_ASSIGNED (\S+)$", out, re.M)
            if len(receipts) == 1:
                return receipts[0]
            raise RuntimeError(f"無法確認 {session} 的遠端識別，配置狀態待確認")


def token_ok() -> bool:
    return (Path.home() / ".config" / "colab-cli" / "token.json").is_file()


def usage(*, account: str | None = None, deadline: float | None = None) -> tuple[float | None, str]:
    account = account or account_scope()
    if not account:
        return None, "無法確認 Colab 登入身分"
    try:
        out = colab("usage", timeout=120, check=False, account=account, deadline=deadline)
    except Exception:
        out = ""
    m = re.search(r"([\d.]+)\s*(?:compute units|運算單元|units)", out, re.I)
    return (float(m.group(1)) if m else None), redact(out).strip() or "無法讀取剩餘運算單元"


def parse_sessions(out: str) -> list[str]:
    names = []
    for line in out.splitlines():
        m = re.match(r"^\[([^\]]+)\]\s+(.+)$", line)
        if not m:
            continue
        name = m.group(1)
        if name in ("?", "colab"):
            continue
        names.append(name)
    return names


def listed_runtimes(out: str) -> dict[str, tuple[str, str]] | None:
    """Return endpoint → (alias, hardware) for a complete listing."""
    rows = [
        line for line in out.strip().splitlines() if line
        and not re.fullmatch(r"\[colab\] Pruned \d+ stale local session\(s\)\.", line)
        and not line.startswith((
            "[colab] A new version of Colab CLI is available: ",
            "[colab] You can run 'colab update --install'",
            "[colab] Run 'colab update' to update.",
            '[colab] To silence this check, set "enable_update_check":',
        ))
    ]
    if rows == ["[colab] No active sessions found on server."]:
        return {}
    assignments = {}
    names = set()
    for row in rows:
        match = re.fullmatch(r"\[([^\]]+)\] (\S+) \| Hardware: (.+)", row)
        if not match:
            return None
        name, endpoint, hardware = match.groups()
        if endpoint in assignments or (name != "?" and name in names):
            return None
        assignments[endpoint] = name, hardware.split(" | ", 1)[0]
        names.add(name)
    return assignments or None


def listed_assignments(out: str) -> dict[str, str] | None:
    """Return endpoint → alias, including unnamed assignments."""
    runtimes = listed_runtimes(out)
    return None if runtimes is None else {ep: name for ep, (name, _) in runtimes.items()}


def sessions(
    *, check: bool = False, account: str | None = None, deadline: float | None = None,
) -> tuple[list[str], str]:
    account = account or account_scope()
    try:
        if not account:
            raise RuntimeError("無法確認 Colab 登入身分")
        out = colab("sessions", timeout=120, check=True, account=account, deadline=deadline).strip()
        if account_scope() != account:
            raise RuntimeError("Colab 登入身分已變更")
        assignments = listed_assignments(out)
        if assignments is None:
            raise RuntimeError("無法解析 Colab Session 清單")
        return [name for name in assignments.values() if name != "?"], out
    except (OSError, RuntimeError, SystemExit):
        if check:
            raise
        return [], "無法確認 Colab Session 清單"


def session_gone(session: str, names: list[str] | None, out: str, endpoint: str | None = None) -> bool:
    assignments = listed_assignments(out) if names is not None else None
    if assignments is None:
        return False
    if endpoint:
        return endpoint not in assignments
    return session not in assignments.values() and "?" not in assignments.values()


def sessions_checked(
    tries: int = 2, *, account: str | None = None, deadline: float | None = None,
) -> tuple[list[str] | None, str]:
    account = account or account_scope()
    out = ""
    for i in range(max(1, tries)):
        try:
            names, out = sessions(check=True, account=account, deadline=deadline)
            if listed_assignments(out) is not None:
                return names, out
        except ColabDeadline:
            return None, ""
        except (OSError, RuntimeError, SystemExit):
            out = ""
        if i + 1 < tries:
            try:
                time.sleep(remaining_time(2, deadline))
            except ColabDeadline:
                break
    return None, out


def sanitize_session_name(session: str) -> str:
    s = str(session or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", s) or s in (".", ".."):
        raise ValueError(f"不合法的 Session 名稱：{session!r}")
    return s


def keep_file(session: str) -> Path:
    return KEEP / f"{sanitize_session_name(session)}.json"


def keep_info(session: str) -> dict | None:
    try:
        d = json.loads(keep_file(session).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError, RecursionError):
        return None


def runtime_endpoint(session: str, *, deadline: float | None = None) -> str | None:
    """Read the endpoint from the pinned Colab CLI's local store without an API call."""
    from colab_cli.state import StateStore

    store = StateStore()
    path = Path(store.path)
    if not path.exists():
        return None
    # The pinned store's public get() waits on its reader lock without a time budget.
    with store._rwlock.read_lock(timeout=remaining_time(5, deadline)):
        with path.open(encoding="utf-8") as stream:
            state = store._load_raw(stream).get(session)
        return state.endpoint if state else None


def is_owned_session(
    name: str, *, account: str | None = None, assignments: dict[str, str] | None = None,
    deadline: float | None = None,
) -> bool:
    try:
        name = sanitize_session_name(name)
    except ValueError:
        return False
    info = keep_info(name) or {}
    account = account or account_scope()
    endpoint = info.get("runtime_endpoint")
    if not account or not endpoint or info.get("created_by") != "owned":
        return False
    if info.get("account_scope", account) != account:
        return False
    if assignments is not None and assignments.get(endpoint) != name:
        return False
    # Legacy ownership requires a positive match in the active account's listing.
    if "account_scope" not in info and assignments is None:
        return False
    try:
        return runtime_endpoint(name, deadline=deadline) == endpoint
    except OSError:
        return False


def owned_sessions() -> list[str]:
    """Return locally registered Studio runtimes, including releases awaiting verification."""
    account = account_scope()
    names = []
    for path in sorted(KEEP.glob("*.json")):
        info = keep_info(path.stem) or {}
        if (account and info.get("created_by") == "owned" and info.get("runtime_endpoint")
                and info.get("account_scope", account) == account):
            names.append(path.stem)
    return names


def has_pending_allocation(account: str) -> bool:
    return any(
        info.get("created_by") == "pending" and info.get("account_scope") == account
        for path in KEEP.glob("*.json") if (info := keep_info(path.stem))
    )


class KeepChanged(RuntimeError):
    """Another local operation changed the Session lease."""


def _session_lock(session: str, purpose: str, timeout: float = 5) -> FileLock:
    return FileLock(keep_file(session).with_suffix(f".{purpose}.lock"), timeout=timeout)


def same_lease(current: dict | None, snapshot: dict | None) -> bool:
    if current is None or snapshot is None:
        return current is snapshot
    watchdog_fields = {"watchdog_pid", "watchdog_ticket", "watchdog_start_until"}
    return ({key: value for key, value in current.items() if key not in watchdog_fields}
            == {key: value for key, value in snapshot.items() if key not in watchdog_fields})


def set_keep(
    session: str,
    expires: float | None,
    pid: int | None = None,
    owner: str | None = None,
    extra: dict | None = None,
    *,
    reserve: bool = False,
    expected: dict | None | object = _ANY_LEASE,
):
    session = sanitize_session_name(session)
    with _session_lock(session, "lease"):
        previous = keep_info(session)
        if expected is not _ANY_LEASE and not same_lease(previous, expected):
            raise KeepChanged(f"Session {session} 的待命紀錄已變更，請重新確認")
        if expected is None and keep_file(session).exists():
            raise KeepChanged(f"無法讀取 Session {session} 的待命紀錄")
        if reserve and keep_file(session).exists():
            raise FileExistsError(keep_file(session))
        old = previous or {}
        if old.get("release_pending") and not (extra and extra.get("release_pending") is True):
            if expected is not _ANY_LEASE:
                raise KeepChanged(f"Session {session} 正在關機")
            return old
        data = dict(old, session=session, expires=expires, watchdog_pid=pid or old.get("watchdog_pid"),
                    created_by=owner or old.get("created_by", "external"))
        if extra:
            data.update(extra)
        data["lease_id"] = uuid.uuid4().hex
        _write_keep(session, data, reserve=reserve)
        return data


def _write_keep(session: str, data: dict, *, reserve: bool = False):
    kf = keep_file(session)
    stream = tempfile.NamedTemporaryFile(dir=KEEP, mode="w", suffix=".tmp", encoding="utf-8", delete=False)
    tmp = Path(stream.name)
    try:
        with stream:
            json.dump(data, stream, ensure_ascii=False)
        if reserve:
            # Install an allocation reservation only if another run has not claimed the alias.
            os.link(tmp, kf)
        else:
            tmp.replace(kf)
    finally:
        tmp.unlink(missing_ok=True)


def drop_keep(session: str, *, expected: dict | None | object = _ANY_LEASE) -> bool:
    with _session_lock(session, "lease"):
        current = keep_info(session)
        if expected is not _ANY_LEASE and not same_lease(current, expected):
            return False
        if expected is None and keep_file(session).exists():
            return False
        keep_file(session).unlink(missing_ok=True)
        return True


class ReleaseState(StrEnum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    FAILED = "failed"


def release_runtime(
    session: str,
    owner: str | None = None,
    endpoint: str | None = None,
    *,
    account: str | None = None,
    owned_only: bool = False,
    watchdog_keep: dict | None = None,
    deadline: float | None = None,
) -> ReleaseState:
    """Request release and verify the assignment independently of the command reply."""
    session = sanitize_session_name(session)
    lock = _session_lock(session, "release", timeout=0)
    try:
        lock.acquire()
    except LockTimeout:
        info = keep_info(session) or {}
        scope = account_scope()
        if (scope and (account is None or account == scope) and info.get("account_scope") == scope
                and info.get("release_pending")):
            return ReleaseState.PENDING
        return ReleaseState.FAILED
    try:
        return _release_runtime(session, owner, endpoint, account, owned_only, watchdog_keep, deadline)
    finally:
        lock.release()


def _release_runtime(session, owner, endpoint, account, owned_only, watchdog_keep, deadline) -> ReleaseState:
    keep = keep_info(session)
    if keep is None and keep_file(session).exists():
        return ReleaseState.FAILED
    if watchdog_keep is not None and not same_lease(keep, watchdog_keep):
        return ReleaseState.PENDING
    info = keep or {}
    supplied_identity = account is not None and endpoint is not None
    account = account or account_scope()
    if not account or account_scope() != account or info.get("account_scope", account) != account:
        return ReleaseState.FAILED
    if (owned_only or watchdog_keep is not None) and info.get("created_by") != "owned":
        return ReleaseState.FAILED
    if info.get("created_by") == "pending" and not (owner == "owned" and endpoint):
        return ReleaseState.FAILED
    explicit_name = owner is None and endpoint is None and not (owned_only or watchdog_keep is not None)
    owner = owner or info.get("created_by", "external")
    registered = info.get("runtime_endpoint")
    endpoint = endpoint or registered
    if not endpoint and (not explicit_name or owner != "external"):
        return ReleaseState.FAILED
    same_record = not (registered and registered != endpoint)
    retrying = watchdog_keep is not None
    already_requested = (
        owner == "owned" and same_record and info.get("release_pending")
        and isinstance(info.get("expires"), (int, float)) and time.time() < info["expires"]
    )

    names, out = sessions_checked(account=account, deadline=deadline)
    assignments = listed_assignments(out) if names is not None else None
    listed = next((ep for ep, alias in (assignments or {}).items() if alias == session), None)
    if explicit_name and registered and listed and listed != registered:
        print(f"⚠️ {session} 的遠端識別已變更，請先確認關機目標", flush=True)
        return ReleaseState.FAILED
    endpoint = endpoint or listed
    # A legacy record can be bound only by a positive alias + endpoint match.
    bound = supplied_identity or info.get("account_scope") == account or (endpoint is not None and listed == endpoint)
    if not endpoint or not bound or account_scope() != account:
        return ReleaseState.FAILED

    def confirm_absence() -> bool:
        if not session_gone(session, names, out, endpoint) or account_scope() != account:
            return False
        if same_record:
            drop_keep(session, expected=keep)
        return True

    if confirm_absence():
        return ReleaseState.CONFIRMED

    scheduled = retrying
    if owner == "owned" and same_record:
        # Commit the shutdown intent against the checked lease before dispatching stop.
        if not already_requested:
            try:
                keep = set_keep(
                    session, time.time() + WATCHDOG_TICK, owner=owner, expected=keep,
                    extra={"busy_until": None, "active_run_id": None, "release_pending": True,
                           "runtime_endpoint": endpoint, "account_scope": account},
                )
            except KeepChanged:
                return ReleaseState.PENDING if retrying else ReleaseState.FAILED
            except OSError as exc:
                print(f"⚠️ 無法登記 {session} 的關機要求（{type(exc).__name__}）", flush=True)
                return ReleaseState.PENDING if retrying else ReleaseState.FAILED
        try:
            if not retrying:
                start_watchdog(session)
            scheduled = True
        except Exception as exc:
            print(f"⚠️ 無法登記 {session} 的關機重試（{type(exc).__name__}）", flush=True)
    if not already_requested and same_record and listed == endpoint and same_lease(keep_info(session), keep):
        try:
            colab("stop", "-s", session, timeout=300, account=account, endpoint=endpoint, deadline=deadline)
        except (OSError, RuntimeError, SystemExit) as exc:
            print(f"⚠️ 無法送出 {session} 的關機請求（{type(exc).__name__}）", flush=True)
        names, out = sessions_checked(account=account, deadline=deadline)
        if confirm_absence():
            return ReleaseState.CONFIRMED
    return ReleaseState.PENDING if scheduled else ReleaseState.FAILED


def _watchdog_locked(session: str) -> bool:
    try:
        with _session_lock(session, "watchdog", timeout=0):
            return False
    except LockTimeout:
        return True


def watchdog_alive(session: str) -> bool:
    try:
        with _session_lock(session, "lease", timeout=0):
            info = keep_info(session)
            return bool(info and not info.get("watchdog_ticket") and _watchdog_locked(session))
    except LockTimeout:
        return False


def remote_exec(
    session: str, code: str, timeout: int = 120, tries: int = 3,
    *, account: str, endpoint: str, deadline: float | None = None,
) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code.rstrip("\n") + "\nprint('CSTUDIO_EXEC_OK')\n")
    try:
        for k in range(tries):
            kernel_timeout = remaining_time(timeout, deadline)
            out = colab("exec", "-s", session, "-f", f.name, "--timeout", kernel_timeout,
                        timeout=timeout + 30, check=False, account=account, endpoint=endpoint, deadline=deadline)
            lines = out.splitlines()
            if "CSTUDIO_EXEC_OK" in lines:
                return "\n".join(line for line in lines if line != "CSTUDIO_EXEC_OK").strip()
            if k + 1 < tries:
                print(f"…exec 沒有回應，重試（{k + 1}/{tries - 1}）", flush=True)
        raise RuntimeError("colab exec 連續沒有回應")
    finally:
        os.unlink(f.name)


def poll_remote_status(
    session: str, timeout: int = 60, *, account: str, endpoint: str, deadline: float | None = None,
    run_id: str | None = None,
) -> dict | None:
    code = vm_program("STATUS", "runtime_status", run_id)
    try:
        out = remote_exec(session, code, timeout=timeout, tries=1, account=account, endpoint=endpoint, deadline=deadline)
    except ColabTimeout:
        raise
    except RuntimeError:
        return None
    for line in reversed(out.splitlines()):
        if line.startswith("CSTUDIO_STATUS "):
            try:
                status = json.loads(line.partition(" ")[2])
                if isinstance(status, dict) and (run_id is None or status.get("run_id") == run_id):
                    return status
            except ValueError:
                return None
    return None


def vm_program(label: str, function: str, *args, include_start: bool = False, root: str | None = None) -> str:
    sources = ["job_paths.py", "cancel.py", "status.py"]
    if include_start:
        sources.append("start.py")
    files = "\n\n".join((REMOTE / name).read_text(encoding="utf-8") for name in sources)
    params = ", ".join(repr(arg) for arg in args)
    source = files + f"\nprint('CSTUDIO_{label} ' + json.dumps({function}(Path({root or REMOTE_ROOT!r}), {params}), ensure_ascii=False))\n"
    return f"exec({source!r}, {{'__name__': 'cstudio_vm', '__package__': None}})"


def active_endpoint(session: str, account: str, *, deadline: float | None = None) -> str:
    """Resolve a selected alias in the active account without adopting stale keep data."""
    names, out = sessions_checked(account=account, deadline=deadline)
    assignments = listed_assignments(out) if names is not None else None
    endpoint = next((ep for ep, alias in (assignments or {}).items() if alias == session), None)
    info = keep_info(session)
    keep = info or {}
    if (
        not endpoint or runtime_endpoint(session, deadline=deadline) != endpoint or account_scope() != account
        or (info is None and keep_file(session).exists())
        or keep.get("account_scope", account) != account or keep.get("runtime_endpoint") not in (None, endpoint)
        or keep.get("created_by") == "pending"
    ):
        raise RuntimeError("無法確認 Colab Session 的帳號及遠端識別")
    return endpoint


def check_recovery_target(session: str, account: str, endpoint: str, *, deadline: float):
    """Recheck local identity and shutdown intent between recovery operations."""
    if account_scope() != account:
        raise RuntimeError("目前的 Colab 登入身分與原批次不符")
    info = keep_info(session) or {}
    if info.get("release_pending") and info.get("runtime_endpoint") == endpoint:
        raise RuntimeError("原 Colab runtime 正在關機")
    if runtime_endpoint(session, deadline=deadline) != endpoint:
        raise RuntimeError("本機 Session 與原 Colab runtime 識別不符")


def verify_recovery_runtime(session: str, account: str, endpoint: str, *, deadline: float):
    """Require positive evidence for the saved assignment before accessing it."""
    if account_scope() != account:
        raise RuntimeError("目前的 Colab 登入身分與原批次不符")
    names, out = sessions_checked(account=account, deadline=deadline)
    assignments = listed_assignments(out) if names is not None else None
    if account_scope() != account:
        raise RuntimeError("目前的 Colab 登入身分與原批次不符")
    if assignments is None:
        raise RuntimeError("無法確認原 Colab runtime 是否仍可用，請稍後重試")
    if endpoint not in assignments:
        raise RuntimeUnavailable("原 Colab runtime 已不存在，無法補抓遠端成果")
    if assignments[endpoint] != session:
        raise RuntimeError("原 Colab runtime 仍存在，但 Session 名稱與原批次不符")
    check_recovery_target(session, account, endpoint, deadline=deadline)


def keep_target(session: str, info: dict, account: str | None) -> dict:
    """Bind a keep update to a verified assignment, preserving proven ownership."""
    if not account:
        raise RuntimeError("無法確認 Colab 登入身分")
    endpoint = active_endpoint(session, account)
    if not same_lease(keep_info(session) or {}, info):
        raise RuntimeError("Colab Session 記錄已變更，請重新確認")
    matched = info.get("runtime_endpoint") == endpoint
    return {
        "owner": "owned" if matched and info.get("created_by") == "owned" else "external",
        "extra": {"account_scope": account, "runtime_endpoint": endpoint,
                  "bundle_hash": info.get("bundle_hash") if matched else None},
    }


def request_cancel(
    session: str, run_id: str | None = None, *, allow_start: bool = False,
    account: str | None = None, endpoint: str | None = None, deadline: float | None = None,
) -> dict:
    session = sanitize_session_name(session)
    if run_id is not None and not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise ValueError("不合法的批次 ID")
    account = account or account_scope()
    if not account:
        raise RuntimeError("無法確認 Colab 登入身分")
    endpoint = endpoint or active_endpoint(session, account, deadline=deadline)
    code = vm_program("CANCEL", "request_run_cancel", run_id, allow_start)
    out = remote_exec(session, code, timeout=15, tries=1, account=account, endpoint=endpoint, deadline=deadline)
    for line in reversed(out.splitlines()):
        if line.startswith("CSTUDIO_CANCEL "):
            result = json.loads(line.partition(" ")[2])
            if (
                isinstance(result, dict)
                and result.get("state") in ("idle", "pending", "confirmed", "finished", "failed", "unsupported")
                and (run_id is None or result.get("run_id") == run_id)
            ):
                return redact_diagnostics(result)
    raise RuntimeError("遠端未回傳可確認的取消狀態")


def start_watchdog(session: str, *, require_ready: bool = False, deadline: float | None = None):
    _launch_watchdog(session)
    if require_ready:
        ready_by = time.monotonic() + remaining_time(WATCHDOG_START_TIMEOUT, deadline)
        try:
            while not watchdog_alive(session):
                time.sleep(remaining_time(0.05, ready_by))
        except ColabDeadline:
            raise OSError(f"無法確認 Session {session} 的 watchdog 已啟動") from None


def _launch_watchdog(session: str):
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200
    else:
        kw["start_new_session"] = True
    with _session_lock(session, "lease"):
        info = keep_info(session)
        if info is None or _watchdog_locked(session):
            return
        now = time.time()
        starting = info.get("watchdog_start_until")
        if isinstance(starting, (float, int)) and now < starting <= now + WATCHDOG_START_TIMEOUT:
            return
        ticket = uuid.uuid4().hex
        launching = dict(info, watchdog_ticket=ticket, watchdog_start_until=now + WATCHDOG_START_TIMEOUT)
        _write_keep(session, launching)
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "cstudio.cli", "_watchdog", "--session", session, "--ticket", ticket],
                env=ENV, **kw,
            )
        except OSError:
            _write_keep(session, info)
            raise
        _write_keep(session, dict(launching, watchdog_pid=child.pid))


def watchdog_loop(session: str, ticket: str | None = None):
    session = sanitize_session_name(session)
    leader = _session_lock(session, "watchdog", timeout=0.5)
    try:
        with _session_lock(session, "lease"):
            info = keep_info(session)
            if info is None or (ticket is not None and info.get("watchdog_ticket") != ticket):
                return
            leader.acquire()
            info.pop("watchdog_ticket", None)
            info.pop("watchdog_start_until", None)
            _write_keep(session, dict(info, watchdog_pid=os.getpid()))
        _monitor_keep(session, leader)
    except LockTimeout:
        return
    finally:
        leader.release()


def _monitor_keep(session: str, leader: FileLock):
    log_path = KEEP / f"{session}.log"

    def say(msg: str):
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")
        except Exception:
            pass

    def finite(x) -> bool:
        try:
            return math.isfinite(x)
        except OverflowError:
            return False

    def exit_if_unchanged(snapshot: dict | None) -> bool:
        # Coordinate exit with start_watchdog in case the alias just got a new lease.
        with _session_lock(session, "lease"):
            if not same_lease(keep_info(session), snapshot):
                return False
            leader.release()
            return True

    say("watchdog started")
    loops = 0
    bad_since = None
    while True:
        time.sleep(WATCHDOG_TICK)
        loops += 1
        try:
            if not keep_file(session).exists():
                if exit_if_unchanged(None):
                    say("keep file removed: exit")
                    return
                continue
            info = keep_info(session)
            if info is None:
                say("cannot read keep record; waiting for ownership evidence")
                continue
            account = account_scope()
            if not account or info.get("account_scope", account) != account:
                continue
            deadline = time.monotonic() + RECOVERY_TIMEOUT
            endpoint = info.get("runtime_endpoint")
            if info.get("created_by") == "pending":
                # An interrupted allocation can lack a local name-to-endpoint mapping.
                # Confirm absence when possible, but never stop an unclaimed runtime.
                if "account_scope" not in info:
                    say("allocation is unconfirmed; inspect Colab runtimes for an unmatched assignment")
                    if exit_if_unchanged(info):
                        return
                    continue
                if loops % 10 == 0 and time.time() >= (info.get("busy_until") or 0):
                    names, lout = sessions_checked(account=account, deadline=deadline)
                    assignments = listed_assignments(lout) if names is not None else None
                    if (
                        assignments is not None and session not in assignments.values()
                        and assignments.keys() <= set(info.get("allocation_baseline", []))
                        and account_scope() == account and drop_keep(session, expected=info)
                    ):
                        say("no active assignment found for pending allocation")
                        if exit_if_unchanged(None):
                            return
                        continue
                    say("allocation is unconfirmed; inspect Colab runtimes for an unmatched assignment")
                continue
            if loops % 10 == 0 and info.get("account_scope") == account and endpoint:
                names, lout = sessions_checked(account=account, deadline=deadline)
                if (session_gone(session, names, lout, endpoint) and account_scope() == account
                        and drop_keep(session, expected=info)):
                    say("session already gone: dropping keep file")
                    if exit_if_unchanged(None):
                        return
                    continue
            exp = info.get("expires")
            busy_until = info.get("busy_until")
            now = time.time()
            has_busy_limit = isinstance(busy_until, (int, float)) and finite(busy_until)
            hard_expired = has_busy_limit and now >= busy_until
            release_pending = bool(info.get("release_pending"))
            if has_busy_limit and not hard_expired and not release_pending:
                continue
            bad = not (
                exp is None
                or (isinstance(exp, (int, float)) and not isinstance(exp, bool) and (finite(exp) or exp == float("-inf")))
            )
            if bad and not hard_expired:
                if bad_since is None:
                    bad_since = now
                if now - bad_since < 30 * 60:
                    continue
                say("keep file unreadable for 30 min: treating as expired")
            else:
                bad_since = None
                if not hard_expired and (exp is None or now < exp):
                    continue

            if info.get("created_by") != "owned":
                if info.get("active_run_id"):
                    continue
                if drop_keep(session, expected=info):
                    say("external session expired: dropping keep file without stopping VM")
                    if exit_if_unchanged(None):
                        return
                continue

            if not endpoint:
                say("keep record has no runtime endpoint; ownership is unconfirmed")
                if exit_if_unchanged(info):
                    return
                continue
            if not (hard_expired or release_pending or bad) and now < exp + IDLE_ACTIVE_GRACE:
                if "account_scope" not in info and active_endpoint(session, account, deadline=deadline) != endpoint:
                    continue
                st = poll_remote_status(session, account=account, endpoint=endpoint, deadline=deadline)
                if st is None or st.get("alive"):
                    say("idle time expired; waiting briefly for the active or unconfirmed batch")
                    continue

            say("time is up: releasing GPU")
            if release_runtime(session, watchdog_keep=info, account=account, deadline=deadline) == ReleaseState.CONFIRMED:
                say("confirmed released")
                if exit_if_unchanged(None):
                    return
                continue
            say("release unconfirmed; will verify on next tick")
        except Exception as e:
            say(f"error (will retry): {type(e).__name__}: {redact(str(e))[:200]}")
