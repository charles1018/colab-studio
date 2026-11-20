# -*- coding: utf-8 -*-
"""Two-step Google login for google-colab-cli:
  uv run cstudio login url            -> prints the sign-in URL
  uv run cstudio login code          -> reads the code securely and saves the Colab CLI token
  uv run cstudio login status
"""
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from getpass import GetPassWarning, getpass
from importlib import resources
from pathlib import Path

from filelock import FileLock, Timeout as LockTimeout

CONF = Path.home() / ".config" / "colab-cli"
TOKEN = CONF / "token.json"
PENDING = CONF / "pending_login.json"
LOGIN_TTL = 10 * 60
MAX_WEB_FLOWS = 10
STATE_INVALID = "登入狀態無效、已過期或已使用，請重新取得登入網址"
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REDIRECT = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"


class LoginError(RuntimeError):
    """A fixed OAuth error suitable for the CLI and Web UI."""


def credential_scope(data: dict) -> str | None:
    """Identify the local OAuth grant across access-token refreshes."""
    grant = [data.get("client_id"), data.get("refresh_token")]
    if not all(isinstance(value, str) and value for value in grant):
        return None
    return hashlib.sha256(json.dumps(grant).encode()).hexdigest()


def account_scope() -> str | None:
    try:
        data = json.loads(TOKEN.read_text(encoding="utf-8"))
        return credential_scope(data) if isinstance(data, dict) else None
    except (OSError, ValueError, RecursionError):
        return None


def client_config():
    return json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())


@contextmanager
def _login_lock():
    try:
        CONF.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(PENDING.with_suffix(".lock"), timeout=5, mode=0o600):
            yield
    except LockTimeout:
        raise LoginError("另一個 Colab 登入正在處理，請稍後再試") from None
    except OSError:
        raise LoginError("無法存取本機 Colab 登入檔案") from None


@contextmanager
def _quiet_oauth_logs():
    # requests-oauthlib DEBUG records contain the full request and token response.
    logger = logging.getLogger("requests_oauthlib.oauth2_session")

    def discard(_record):
        return False

    logger.addFilter(discard)
    try:
        yield
    finally:
        logger.removeFilter(discard)


def _write_private(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}-", encoding="utf-8", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if os.name == "posix":
                os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pending_flows(now: float) -> tuple[dict, dict | None]:
    try:
        data = json.loads(PENDING.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, RecursionError):
        return {}, None
    # Previous pending records had neither an expiry nor a CLI/Web distinction.
    if not isinstance(data, dict) or data.get("version") != 1:
        return {}, None

    def valid(record):
        return (
            isinstance(record, dict)
            and all(isinstance(record.get(key), str) and record[key] for key in ("state", "verifier"))
            and isinstance(record.get("created_at"), (int, float))
            and now - LOGIN_TTL < record["created_at"] <= now
        )

    web = data.get("web")
    cli = data.get("cli")
    active = {
        state: record for state, record in web.items() if valid(record) and record["state"] == state
    } if isinstance(web, dict) else {}
    return active, cli if valid(cli) else None


def _save_flows(web: dict, cli: dict | None) -> None:
    if web or cli:
        _write_private(PENDING, json.dumps({"version": 1, "web": web, "cli": cli}))
    else:
        PENDING.unlink(missing_ok=True)


def get_login_url(*, cli: bool = False) -> tuple[str, str]:
    """Create an expiring verifier for a Web tab or the single CLI login."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    with _login_lock():
        try:
            with _quiet_oauth_logs():
                flow = InstalledAppFlow.from_client_config(client_config(), SCOPES, autogenerate_code_verifier=True)
                flow.redirect_uri = REDIRECT
                url, state = flow.authorization_url(prompt="consent", token_usage="remote")
            if not all(isinstance(value, str) and value for value in (url, state, flow.code_verifier)):
                raise ValueError("incomplete flow")
        except Exception:
            raise LoginError("無法產生 Google OAuth 登入網址") from None
        now = time.time()
        web, pending_cli = _pending_flows(now)
        record = {"state": state, "verifier": flow.code_verifier, "created_at": now}
        if cli:
            pending_cli = record
        else:
            web[state] = record
            web = dict(list(web.items())[-MAX_WEB_FLOWS:])
        _save_flows(web, pending_cli)
        return url, state


def submit_login_code(auth_code: str, state: str | None = None) -> Path:
    """Consume one Web state or the CLI's pending flow and replace the shared token."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    code = auth_code.strip() if isinstance(auth_code, str) else ""
    if not code:
        raise LoginError("請貼上 Google 授權碼")
    if state is not None and (not isinstance(state, str) or not state):
        raise LoginError(STATE_INVALID)
    with _login_lock():
        web, pending_cli = _pending_flows(time.time())
        chosen = pending_cli if state is None else web.pop(state, None)
        if chosen is None:
            raise LoginError(STATE_INVALID)
        # Reserve the attempt before sending the code, including cases with a lost exchange reply.
        _save_flows(web, None if state is None else pending_cli)
        try:
            with _quiet_oauth_logs():
                flow = InstalledAppFlow.from_client_config(
                    client_config(), SCOPES, state=chosen["state"],
                    code_verifier=chosen["verifier"], autogenerate_code_verifier=False,
                )
                flow.redirect_uri = REDIRECT
                flow.fetch_token(code=code, timeout=30)
                token_json = flow.credentials.to_json()
                info = json.loads(token_json)
                required = ("token", "refresh_token", "client_id", "client_secret")
                if not isinstance(info, dict) or not all(isinstance(info.get(k), str) and info[k] for k in required):
                    raise ValueError("incomplete credentials")
        except Exception:
            raise LoginError("Google OAuth 交換失敗，請重新取得登入網址") from None
        _write_private(TOKEN, token_json)
        return TOKEN


def handle_login(action: str = "status", auth_code: str | None = None):
    if action == "status":
        print("✅ 已登入 (" + str(TOKEN) + ")" if TOKEN.is_file() else "❌ 尚未登入")
    elif action == "url":
        try:
            url, _ = get_login_url(cli=True)
        except LoginError as exc:
            sys.exit(str(exc))
        print("請用【有 Google AI Pro 訂閱】的帳號打開這個網址登入，按允許後，把頁面上顯示的授權碼貼回來：")
        print(url)
    elif action == "code":
        try:
            if auth_code is None:
                if sys.stdin.isatty():
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", GetPassWarning)
                        auth_code = getpass("請貼上 Google 授權碼：")
                else:
                    auth_code = sys.stdin.readline()
            submit_login_code(auth_code)
        except (EOFError, GetPassWarning, OSError):
            raise SystemExit("無法安全讀取授權碼") from None
        except LoginError as exc:
            raise SystemExit(f"登入失敗：{exc}") from None
        print("✅ Colab 登入完成")
    else:
        sys.exit("用法：uv run cstudio login <status|url|code>")
