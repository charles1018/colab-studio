"""Offline regression tests for expiring, private OAuth login state."""
import io
import itertools
import json
import logging
import os
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.oauth2.credentials import Credentials

from cstudio import colab_login as L


CODE = "fixture-authorization-code"
ACCESS = "fixture-access-token"
REFRESH = "fixture-refresh-token"
VERIFIER = "fixture-pkce-verifier"
LIBRARY_LOGGER = logging.getLogger("requests_oauthlib.oauth2_session")


class OAuthLoginTests(unittest.TestCase):
    def setUp(self):
        self.conf = Path(self.enterContext(tempfile.TemporaryDirectory())) / "colab-cli"
        self.token = self.conf / "token.json"
        self.pending = self.conf / "pending_login.json"
        self.now = 10_000.0
        self.sequence = itertools.count(1)
        self.payload = {
            "token": ACCESS, "refresh_token": REFRESH,
            "client_id": "fixture-client", "client_secret": "fixture-client-secret",
        }
        self.exchange = Mock()
        self.authorization_log = Mock()
        self.enterContext(patch.multiple(
            L, CONF=self.conf, TOKEN=self.token, PENDING=self.pending,
            time=SimpleNamespace(time=lambda: self.now), client_config=Mock(return_value={}),
        ))
        self.factory = self.enterContext(patch(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_config", side_effect=self.flow,
        ))

    def flow(self, config, scopes, *, autogenerate_code_verifier, state=None, code_verifier=None):
        self.assertEqual(scopes, L.SCOPES)
        if autogenerate_code_verifier:
            state = f"fixture-state-{next(self.sequence)}"
            code_verifier = f"{VERIFIER}-{state}"
        flow = SimpleNamespace(code_verifier=code_verifier)

        def authorization_url(**options):
            self.assertEqual(flow.redirect_uri, L.REDIRECT)
            self.assertEqual(options, {"prompt": "consent", "token_usage": "remote"})
            self.authorization_log()
            return f"https://example.test/oauth?state={state}", state

        def fetch_token(*, code, timeout):
            self.assertEqual(flow.redirect_uri, L.REDIRECT)
            self.assertEqual(timeout, 30)
            self.exchange(state, code_verifier, code)

        flow.authorization_url = authorization_url
        flow.fetch_token = fetch_token
        flow.credentials = SimpleNamespace(to_json=lambda: json.dumps(self.payload))
        return flow

    def seed_previous_token(self):
        self.conf.mkdir(exist_ok=True)
        content = json.dumps(dict(self.payload, token="old-access", refresh_token="old-refresh"))
        self.token.write_text(content, encoding="utf-8")
        return content

    def test_exact_web_state_isolated_from_cli_and_other_tabs(self):
        old = self.seed_previous_token()
        _, first = L.get_login_url()
        _, second = L.get_login_url()
        for wrong in (None, "", "unknown", f" {second}"):
            with self.subTest(state=wrong), self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
                L.submit_login_code(CODE, state=wrong)
        self.exchange.assert_not_called()
        self.assertEqual(self.token.read_text(encoding="utf-8"), old)

        _, cli = L.get_login_url(cli=True)
        with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
            L.submit_login_code(CODE, state=cli)
        self.assertEqual(L.submit_login_code(CODE, state=first), self.token)
        self.exchange.assert_called_once_with(first, f"{VERIFIER}-{first}", CODE)
        with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
            L.submit_login_code(CODE, state=first)
        self.assertEqual(set(json.loads(self.pending.read_text())["web"]), {second})
        L.submit_login_code(CODE)
        self.exchange.assert_called_with(cli, f"{VERIFIER}-{cli}", CODE)
        L.submit_login_code(CODE, state=second)
        self.exchange.assert_called_with(second, f"{VERIFIER}-{second}", CODE)
        self.assertFalse(self.pending.exists())

    def test_cli_new_url_replaces_only_its_own_pending_flow(self):
        _, web = L.get_login_url()
        _, first = L.get_login_url(cli=True)
        _, latest = L.get_login_url(cli=True)
        L.submit_login_code(CODE)
        self.exchange.assert_called_once_with(latest, f"{VERIFIER}-{latest}", CODE)
        data = json.loads(self.pending.read_text())
        self.assertIsNone(data["cli"])
        self.assertEqual(set(data["web"]), {web})
        self.assertNotIn(first, self.pending.read_text())

    def test_web_flow_limit_evicts_oldest_and_keeps_cli_state(self):
        _, cli = L.get_login_url(cli=True)
        states = [L.get_login_url()[1] for _ in range(L.MAX_WEB_FLOWS + 1)]
        data = json.loads(self.pending.read_text())
        self.assertEqual(list(data["web"]), states[1:])
        self.assertEqual(data["cli"]["state"], cli)
        with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
            L.submit_login_code(CODE, state=states[0])
        self.exchange.assert_not_called()

    def test_expired_or_future_flows_and_legacy_pending_cannot_exchange(self):
        old = self.seed_previous_token()
        _, state = L.get_login_url()
        L.get_login_url(cli=True)
        self.now += L.LOGIN_TTL
        for origin in (state, None):
            with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
                L.submit_login_code(CODE, state=origin)
        _, future = L.get_login_url()
        self.now -= 1
        with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
            L.submit_login_code(CODE, state=future)
        self.pending.write_text(json.dumps({"state": state, "verifier": VERIFIER}), encoding="utf-8")
        for origin in (state, None):
            with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
                L.submit_login_code(CODE, state=origin)
        self.exchange.assert_not_called()
        self.assertEqual(self.token.read_text(encoding="utf-8"), old)

    def test_concurrent_exchange_of_the_same_state_reaches_provider_once(self):
        _, state = L.get_login_url()
        exchanging = Event()
        release = Event()

        def hold(*_args):
            exchanging.set()
            self.assertTrue(release.wait(3))

        self.exchange.side_effect = hold
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(L.submit_login_code, CODE, state)
            try:
                self.assertTrue(exchanging.wait(3))
                self.assertFalse(self.pending.exists())
                repeat = pool.submit(L.submit_login_code, CODE, state)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=3), self.token)
            with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
                repeat.result(timeout=3)
        self.exchange.assert_called_once()

    def test_provider_diagnostics_are_private_and_failed_attempt_is_consumed(self):
        old = self.seed_previous_token()
        _, state = L.get_login_url()
        _, other = L.get_login_url()
        secret_text = f"{CODE} {VERIFIER} {ACCESS} {REFRESH}"

        def provider_error(*_args):
            LIBRARY_LOGGER.debug("provider payload: %s", secret_text)
            raise RuntimeError(secret_text)

        self.exchange.side_effect = provider_error
        with self.assertLogs(LIBRARY_LOGGER, level="DEBUG") as logs:
            with self.assertRaisesRegex(L.LoginError, "Google OAuth 交換失敗") as failed:
                L.submit_login_code(CODE, state=state)
            LIBRARY_LOGGER.debug("after exchange")
        self.assertEqual([record.getMessage() for record in logs.records], ["after exchange"])
        self.assertNotIn(secret_text, str(failed.exception))
        self.assertEqual(self.token.read_text(encoding="utf-8"), old)
        with self.assertRaisesRegex(L.LoginError, "登入狀態無效"):
            L.submit_login_code(CODE, state=state)
        self.exchange.assert_called_once()
        self.assertEqual(set(json.loads(self.pending.read_text())["web"]), {other})

    def test_provider_error_while_creating_url_does_not_expose_config(self):
        self.authorization_log.side_effect = RuntimeError(VERIFIER)
        with self.assertRaisesRegex(L.LoginError, "無法產生 Google OAuth 登入網址") as failed:
            L.get_login_url()
        self.assertNotIn(VERIFIER, str(failed.exception))
        self.assertFalse(self.pending.exists())

    def test_bad_token_or_interrupted_replace_preserves_previous_credentials(self):
        old = self.seed_previous_token()
        _, state = L.get_login_url()
        self.payload["refresh_token"] = None
        with self.assertRaisesRegex(L.LoginError, "Google OAuth 交換失敗"):
            L.submit_login_code(CODE, state=state)
        self.payload["refresh_token"] = REFRESH
        self.assertEqual(self.token.read_text(encoding="utf-8"), old)
        _, state = L.get_login_url()
        replace_file = Path.replace

        def replace(source, dest):
            if dest == self.token:
                self.assertEqual(source.parent, self.token.parent)
                raise PermissionError(ACCESS)
            return replace_file(source, dest)

        with patch.object(Path, "replace", replace), self.assertRaisesRegex(L.LoginError, "無法存取") as failed:
            L.submit_login_code(CODE, state=state)
        self.assertNotIn(ACCESS, str(failed.exception))
        self.assertEqual(self.token.read_text(encoding="utf-8"), old)
        self.assertFalse(list(self.conf.glob(".token.json-*")))

    @unittest.skipUnless(os.name == "posix", "POSIX mode bits")
    def test_private_credentials_and_pending_files_under_permissive_umask(self):
        original = os.umask(0)
        try:
            _, first = L.get_login_url()
            _, second = L.get_login_url()
            for path, mode in ((self.conf, 0o700), (self.pending, 0o600)):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode)
            L.submit_login_code(CODE, first)
            self.assertEqual(stat.S_IMODE(self.token.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(self.pending.stat().st_mode), 0o600)
            self.assertNotIn(first, self.pending.read_text())
            self.assertIn(second, self.pending.read_text())
        finally:
            os.umask(original)
        credentials = Credentials.from_authorized_user_file(str(self.token), L.SCOPES)
        self.assertEqual(credentials.refresh_token, REFRESH)
        self.assertEqual(L.account_scope(), L.credential_scope(self.payload))

    def test_cli_piped_hidden_and_legacy_inputs_reach_only_the_cli_flow(self):
        with redirect_stdout(io.StringIO()) as output:
            for source in ("pipe", "tty", "legacy"):
                L.handle_login("url")
                pending_cli = json.loads(self.pending.read_text())["cli"]
                stdin = io.StringIO(CODE + "\n")
                with (
                    patch.object(L.sys, "stdin", stdin),
                    patch.object(stdin, "isatty", return_value=source == "tty"),
                    patch.object(L, "getpass", return_value=CODE) as hidden,
                ):
                    L.handle_login("code", CODE if source == "legacy" else None)
                self.assertEqual(hidden.call_count, int(source == "tty"))
                self.exchange.assert_called_with(pending_cli["state"], pending_cli["verifier"], CODE)
        for secret in (CODE, VERIFIER, ACCESS, REFRESH):
            self.assertNotIn(secret, output.getvalue())
