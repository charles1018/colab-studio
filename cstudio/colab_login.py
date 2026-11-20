# -*- coding: utf-8 -*-
"""Two-step Google login for google-colab-cli:
  uv run cstudio login url            -> prints the sign-in URL
  uv run cstudio login code <CODE>    -> saves OAuth token to ~/.config/colab-cli/token.json
  uv run cstudio login status
"""
import hashlib
import json
import sys
from importlib import resources
from pathlib import Path

CONF = Path.home() / ".config" / "colab-cli"
TOKEN = CONF / "token.json"
PENDING = CONF / "pending_login.json"
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
REDIRECT = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"


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


def get_login_url() -> tuple[str, str]:
    """Generate Google OAuth authorization URL and persist pending verifier/state.
    Returns (url, state).
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    CONF.mkdir(parents=True, exist_ok=True)
    flow = InstalledAppFlow.from_client_config(client_config(), SCOPES, autogenerate_code_verifier=True)
    flow.redirect_uri = REDIRECT
    url, state = flow.authorization_url(prompt="consent", token_usage="remote")
    pend_map = {}
    if PENDING.is_file():
        try:
            old = json.loads(PENDING.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                pend_map = old.get("flows") if isinstance(old.get("flows"), dict) else {}
        except Exception:
            pend_map = {}
    pend_map[state] = {"verifier": flow.code_verifier, "state": state}
    # Keep at most 10 recent login flows so multi-tab login won't overwrite each other
    if len(pend_map) > 10:
        for k in list(pend_map.keys())[:-10]:
            pend_map.pop(k, None)
    data = {"verifier": flow.code_verifier, "state": state, "flows": pend_map}
    tmp = PENDING.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(PENDING)
    return url, state


def submit_login_code(auth_code: str, state: str | None = None) -> Path:
    """Exchange authorization code for OAuth token and save atomically to TOKEN."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    code = (auth_code or "").strip()
    if not code:
        raise ValueError("請輸入 Google 授權碼")
    if not PENDING.exists():
        raise RuntimeError("找不到待驗證的登入狀態，請先取得登入網址")
    pend = json.loads(PENDING.read_text(encoding="utf-8"))
    flows = pend.get("flows") if isinstance(pend.get("flows"), dict) else {}
    chosen = flows.get(state) if (state and state in flows) else pend
    verifier = chosen.get("verifier")
    st = chosen.get("state")
    if not verifier:
        raise RuntimeError("登入驗證狀態無效，請重新取得登入網址")
    flow = InstalledAppFlow.from_client_config(
        client_config(),
        SCOPES,
        state=st,
        code_verifier=verifier,
        autogenerate_code_verifier=False,
    )
    flow.redirect_uri = REDIRECT
    flow.fetch_token(code=code)
    CONF.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN.with_suffix(".json.tmp")
    tmp.write_text(flow.credentials.to_json(), encoding="utf-8")
    tmp.replace(TOKEN)
    if state and state in flows:
        flows.pop(state, None)
    if flows:
        pend["flows"] = flows
        tmp_p = PENDING.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(pend), encoding="utf-8")
        tmp_p.replace(PENDING)
    else:
        PENDING.unlink(missing_ok=True)
    return TOKEN


def handle_login(action: str = "status", auth_code: str | None = None):
    CONF.mkdir(parents=True, exist_ok=True)
    if action == "status":
        print("✅ 已登入 (" + str(TOKEN) + ")" if TOKEN.is_file() else "❌ 尚未登入")
    elif action == "url":
        url, _ = get_login_url()
        print("請用【有 Google AI Pro 訂閱】的帳號打開這個網址登入，按允許後，把頁面上顯示的授權碼貼回來：")
        print(url)
    elif action == "code":
        if not auth_code:
            sys.exit("用法：uv run cstudio login code <授權碼>")
        if not PENDING.exists():
            sys.exit("請先執行：uv run cstudio login url")
        try:
            tok = submit_login_code(auth_code)
        except Exception as e:
            sys.exit(f"登入失敗：{e}")
        print("✅ 登入完成，token 存在", tok)
    else:
        sys.exit("用法：uv run cstudio login <status|url|code>")
