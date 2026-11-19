# -*- coding: utf-8 -*-
"""Two-step Google login for google-colab-cli:
  uv run cstudio login url            -> prints the sign-in URL
  uv run cstudio login code <CODE>    -> saves OAuth token to ~/.config/colab-cli/token.json
  uv run cstudio login status
"""
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


def client_config():
    return json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())


def handle_login(action: str = "status", auth_code: str | None = None):
    from google_auth_oauthlib.flow import InstalledAppFlow

    CONF.mkdir(parents=True, exist_ok=True)
    if action == "status":
        print("✅ 已登入 (" + str(TOKEN) + ")" if TOKEN.is_file() else "❌ 尚未登入")
    elif action == "url":
        flow = InstalledAppFlow.from_client_config(client_config(), SCOPES, autogenerate_code_verifier=True)
        flow.redirect_uri = REDIRECT
        url, state = flow.authorization_url(prompt="consent", token_usage="remote")
        PENDING.write_text(json.dumps({"verifier": flow.code_verifier, "state": state}), encoding="utf-8")
        print("請用【有 Google AI Pro 訂閱】的帳號打開這個網址登入，按允許後，把頁面上顯示的授權碼貼回來：")
        print(url)
    elif action == "code":
        if not auth_code:
            sys.exit("用法：uv run cstudio login code <授權碼>")
        if not PENDING.exists():
            sys.exit("請先執行：uv run cstudio login url")
        pend = json.loads(PENDING.read_text(encoding="utf-8"))
        flow = InstalledAppFlow.from_client_config(
            client_config(),
            SCOPES,
            state=pend["state"],
            code_verifier=pend["verifier"],
            autogenerate_code_verifier=False,
        )
        flow.redirect_uri = REDIRECT
        flow.fetch_token(code=auth_code.strip())
        TOKEN.write_text(flow.credentials.to_json(), encoding="utf-8")
        PENDING.unlink(missing_ok=True)
        print("✅ 登入完成，token 存在", TOKEN)
    else:
        sys.exit("用法：uv run cstudio login <status|url|code>")
