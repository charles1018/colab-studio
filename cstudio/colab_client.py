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
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = ROOT / "remote"
GPUS = ("L4", "G4", "A100", "H100", "T4")
PREFIX = "cstudio-"
ASSIGN_TIMEOUT = 900
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
KEEP = Path.home() / ".config" / "colab-studio" / "keep"


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


def colab(*args, timeout: int = 600, check: bool = True) -> str:
    try:
        p = subprocess.run(
            [colab_exe(), *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=ENV,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        if check:
            raise ColabTimeout(f"colab {' '.join(map(str, args))} 超過 {timeout} 秒沒有回應") from None
        return ""
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"colab {' '.join(map(str, args))} 失敗（{p.returncode}）：\n{out[-2000:]}")
    return out


def new_session_name() -> str:
    return f"{PREFIX}{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def new_runtime(
    session: str,
    gpu: str,
    retries: int = 5,
    wait: int = 120,
    check_abort: Callable[[], None] | None = None,
    deadline: float | None = None,
) -> str:
    for attempt in range(retries + 1):
        if check_abort:
            check_abort()
        if deadline is not None and time.time() >= deadline:
            raise RuntimeError("已到達 Colab GPU 配置時間上限")
        try:
            # Let a request already in flight return its identity before cleanup.
            return colab("new", "-s", session, "--gpu", gpu, timeout=ASSIGN_TIMEOUT)
        except ColabTimeout:
            raise
        except RuntimeError as e:
            busy = any(k in str(e) for k in ("Service Unavailable", "503", "Too Many Requests", "429"))
            if busy and runtime_endpoint(session):
                # A recorded assignment must be reconciled before any further allocation.
                raise
            if not busy or attempt == retries:
                if busy:
                    raise RuntimeError(
                        f"分配不到 {gpu}（試了 {attempt + 1} 次都是 Service Unavailable）。"
                        f"晚點再試，或改用另一種 GPU（L4／G4）。\n{e}"
                    )
                raise
            print(
                f"⏳ Colab 現在分配不到 {gpu}（Service Unavailable），{wait:g} 秒後再試（第 {attempt + 1}/{retries} 次重試）",
                flush=True,
            )
            if check_abort:
                check_abort()
            delay = min(wait, max(0, deadline - time.time())) if deadline is not None else wait
            time.sleep(delay)
    return ""


def token_ok() -> bool:
    return (Path.home() / ".config" / "colab-cli" / "token.json").is_file()


def usage() -> tuple[float | None, str]:
    try:
        out = colab("usage", timeout=120, check=False)
    except Exception as e:
        out = f"（讀取失敗：{e}）"
    m = re.search(r"([\d.]+)\s*(?:compute units|運算單元|units)", out, re.I)
    return (float(m.group(1)) if m else None), out.strip()


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


def sessions() -> tuple[list[str], str]:
    out = colab("sessions", timeout=120, check=False)
    return parse_sessions(out), out.strip()


def listing_answered(names: list[str] | None, out: str) -> bool:
    return bool(names) or "No active sessions" in out or bool(re.search(r"^\[\?\]", out, re.M))


def session_gone(session: str, names: list[str] | None, out: str) -> bool:
    if names is None or not listing_answered(names, out) or re.search(r"^\[\?\]", out, re.M):
        return False
    return session not in names


def sessions_checked(tries: int = 2) -> tuple[list[str] | None, str]:
    out = ""
    for i in range(max(1, tries)):
        names, out = sessions()
        if listing_answered(names, out):
            return names, out
        if i + 1 < tries:
            time.sleep(2)
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


def runtime_endpoint(session: str) -> str | None:
    """Read the endpoint from the pinned Colab CLI's local store without an API call."""
    from colab_cli.state import StateStore

    state = StateStore().get(session)
    return state.endpoint if state else None


def is_owned_session(name: str) -> bool:
    try:
        name = sanitize_session_name(name)
    except ValueError:
        return False
    info = keep_info(name) or {}
    if info.get("created_by") in ("external", "pending"):
        return False
    if endpoint := info.get("runtime_endpoint"):
        return runtime_endpoint(name) == endpoint
    if name.startswith(PREFIX):
        return True
    if not keep_file(name).exists():
        return False
    return True


def owned_sessions() -> list[str]:
    return [n for n in sessions()[0] if is_owned_session(n)]


def set_keep(
    session: str,
    expires: float | None,
    pid: int | None = None,
    owner: str | None = None,
    extra: dict | None = None,
):
    session = sanitize_session_name(session)
    KEEP.mkdir(parents=True, exist_ok=True)
    old = keep_info(session) or {}
    owner = owner or old.get("created_by")
    if owner is None and not keep_file(session).exists():
        owner = "owned" if session.startswith(PREFIX) else "external"
    data = dict(old, session=session, expires=expires, watchdog_pid=pid or old.get("watchdog_pid"))
    if owner:
        data["created_by"] = owner
    if extra:
        data.update(extra)
    kf = keep_file(session)
    stream = tempfile.NamedTemporaryFile(dir=KEEP, mode="w", suffix=".tmp", encoding="utf-8", delete=False)
    tmp = Path(stream.name)
    try:
        with stream:
            json.dump(data, stream, ensure_ascii=False)
        tmp.replace(kf)
    finally:
        tmp.unlink(missing_ok=True)


def drop_keep(session: str):
    keep_file(session).unlink(missing_ok=True)


def release_runtime(session: str, owner: str, endpoint: str | None = None) -> bool:
    """Persist a retryable release request before stopping a runner's runtime."""
    try:
        set_keep(
            session,
            time.time() + 30,
            owner=owner,
            extra={"busy_until": None, "release_pending": True, "runtime_endpoint": endpoint},
        )
        start_watchdog(session)
    except Exception as exc:
        print(f"⚠️ 無法登記 {session} 的關機重試（{type(exc).__name__}）", flush=True)

    try:
        if endpoint and runtime_endpoint(session) != endpoint:
            print(f"⚠️ {session} 的遠端識別已變更，將僅查詢關機狀態", flush=True)
        else:
            colab("stop", "-s", session, timeout=300, check=False)
    except Exception as exc:
        print(f"⚠️ 無法送出 {session} 的關機請求（{type(exc).__name__}）", flush=True)

    try:
        names, out = sessions_checked()
        if session_gone(session, names, out):
            drop_keep(session)
            return True
    except Exception as exc:
        print(f"⚠️ 無法確認 {session} 的關機狀態（{type(exc).__name__}）", flush=True)
    return False


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def remote_exec(session: str, code: str, timeout: int = 120, tries: int = 3) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code.rstrip("\n") + "\nprint('CSTUDIO_EXEC_OK')\n")
    try:
        for k in range(tries):
            out = colab("exec", "-s", session, "-f", f.name, "--timeout", timeout, timeout=timeout + 30, check=False)
            if "CSTUDIO_EXEC_OK" in out:
                return out.replace("CSTUDIO_EXEC_OK", "").strip()
            print(f"…exec 沒有回應，重試（{k + 1}/{tries}）", flush=True)
        raise RuntimeError("colab exec 連續沒有回應")
    finally:
        os.unlink(f.name)


def poll_remote_status(session: str, timeout: int = 60) -> dict | None:
    status_py = REMOTE / "status.py"
    out = colab("exec", "-s", session, "-f", status_py, "--timeout", timeout, timeout=timeout + 20, check=False)
    for line in reversed(out.splitlines()):
        if "CSTUDIO_STATUS " in line:
            try:
                return json.loads(line.split("CSTUDIO_STATUS ", 1)[1])
            except Exception:
                return None
    return None


def request_cancel(session: str, run_id: str | None = None, *, allow_start: bool = False) -> dict:
    session = sanitize_session_name(session)
    if run_id is not None and not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise ValueError("不合法的批次 ID")
    keep = keep_info(session) or {}
    endpoint = keep.get("runtime_endpoint")
    if endpoint and runtime_endpoint(session) != endpoint:
        raise RuntimeError("Session 遠端識別已變更，無法確認取消目標")
    source = (REMOTE / "cancel.py").read_text(encoding="utf-8")
    code = source + (
        f"\nprint('CSTUDIO_CANCEL ' + json.dumps("
        f"request_cancel(Path('/content/cstudio'), {run_id!r}, allow_start={allow_start!r})))\n"
    )
    out = remote_exec(session, code, timeout=15, tries=1)
    for line in reversed(out.splitlines()):
        if line.startswith("CSTUDIO_CANCEL "):
            result = json.loads(line.partition(" ")[2])
            if (
                isinstance(result, dict)
                and result.get("state") in ("idle", "pending", "confirmed", "finished", "failed", "unsupported")
                and (run_id is None or result.get("run_id") == run_id)
            ):
                return result
    raise RuntimeError("遠端未回傳可確認的取消狀態")


def start_watchdog(session: str):
    info = keep_info(session) or {}
    if pid_alive(info.get("watchdog_pid")):
        return
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen([sys.executable, "-m", "cstudio.cli", "_watchdog", "--session", session], env=ENV, **kw)
    set_keep(session, info.get("expires"), p.pid)


def watchdog_loop(session: str):
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

    say("watchdog started")
    loops = 0
    bad_since = None
    while True:
        time.sleep(30)
        loops += 1
        try:
            if not keep_file(session).exists():
                say("keep file removed: exit")
                return
            info = keep_info(session)
            if info and info.get("created_by") == "pending":
                # An interrupted allocation can lack a local name-to-endpoint mapping.
                # Confirm absence when possible, but never stop an unclaimed runtime.
                if loops % 10 == 0 and time.time() >= info["busy_until"]:
                    names, lout = sessions_checked()
                    if session_gone(session, names, lout) and keep_info(session) == info:
                        drop_keep(session)
                        say("no active assignment found for pending allocation")
                        return
                    say("allocation is unconfirmed; inspect Colab runtimes for an unmatched assignment")
                continue
            if loops % 10 == 0:
                names, lout = sessions_checked()
                if session_gone(session, names, lout) and keep_info(session) == info:
                    say("session already gone: dropping keep file")
                    drop_keep(session)
                    return
            if info is None:
                say("cannot read keep record; waiting for ownership evidence")
                continue
            exp = info.get("expires")
            busy_until = info.get("busy_until")
            hard_expired = isinstance(busy_until, (int, float)) and time.time() >= busy_until
            release_pending = bool(info.get("release_pending"))
            bad = not (
                exp is None
                or (isinstance(exp, (int, float)) and not isinstance(exp, bool) and (finite(exp) or exp == float("-inf")))
            )
            if bad:
                now = time.time()
                if bad_since is None:
                    bad_since = now
                if now - bad_since < 30 * 60:
                    continue
                say("keep file unreadable for 30 min: treating as expired")
            else:
                bad_since = None
                if not hard_expired and (exp is None or time.time() < exp):
                    continue

            if not is_owned_session(session):
                if info.get("runtime_endpoint"):
                    say("runtime identity cannot be verified; retaining release record")
                    continue
                say("external session expired: dropping keep file without stopping VM")
                drop_keep(session)
                return

            if not (hard_expired or release_pending):
                st = poll_remote_status(session)
                if st and st.get("alive"):
                    say("time is up but batch is running: wait")
                    continue
            if keep_info(session) != info:
                continue

            say("time is up: releasing GPU")
            colab("stop", "-s", session, timeout=300, check=False)
            names, lout = sessions_checked()
            if session_gone(session, names, lout):
                drop_keep(session)
                say("confirmed released")
                return
            say("stop issued; will verify on next tick")
        except Exception as e:
            say(f"error (will retry): {type(e).__name__}: {str(e)[:200]}")
