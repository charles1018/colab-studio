"""Exercise the pinned CLI adapter with local stores and an offline Colab client."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from colab_cli import common
from colab_cli.client import Accelerator, AssignmentVariant, Shape
from colab_cli.commands import session
from colab_cli.state import SessionState, StateStore

from cstudio import _colab, colab_login


class ColabIdentityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.token = self.root / "token.json"
        self.grant = {"client_id": "client", "client_secret": "fixture-secret", "refresh_token": "first-account"}
        self.token.write_text(json.dumps(self.grant), encoding="utf-8")
        self.store = StateStore(str(self.root / "sessions.json"))
        self.state = common.State()
        self.state._store = self.store
        self.state._history = Mock()
        self.client = Mock()
        self.runtime = Mock()
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(common, "state", self.state))
        self.enterContext(patch.multiple(_colab.cli, state=self.state, setup_logging=Mock()))
        self.enterContext(patch.object(_colab.cli.auto_update, "run_background_check"))
        self.enterContext(patch.multiple(_colab, TOKEN=self.token, Client=Mock(return_value=self.client)))
        self.enterContext(patch.object(session, "ColabRuntime", return_value=self.runtime))

    def execute(self, *args, endpoint=None):
        env = {"CSTUDIO_COLAB_ACCOUNT_SCOPE": colab_login.credential_scope(self.grant)}
        if endpoint:
            env["CSTUDIO_COLAB_ENDPOINT"] = endpoint
        with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", ["colab", *args]):
            with self.assertRaises(SystemExit) as result:
                _colab.main()
        return result.exception.code

    def register(self, endpoint):
        self.store.add(SessionState(
            name="shared", endpoint=endpoint, token="fixture-proxy-token", url="https://example.invalid",
            token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        ))

    def test_scope_survives_access_token_refresh_and_changes_with_grant(self):
        scope = colab_login.credential_scope(self.grant)
        self.assertEqual(scope, colab_login.credential_scope(dict(self.grant, token="refreshed")))
        self.assertNotEqual(scope, colab_login.credential_scope(dict(self.grant, refresh_token="second-account")))
        self.assertIsNone(colab_login.credential_scope({"token": "access-token-only"}))

    def test_changed_grant_rejected_before_colab_command(self):
        self.token.write_text(json.dumps(dict(self.grant, refresh_token="second-account")), encoding="utf-8")
        self.assertIn("have changed", self.execute("sessions"))
        _colab.Client.assert_not_called()

    def test_listing_neither_prunes_foreign_alias_nor_hides_unnamed_endpoint(self):
        self.register("first-account-endpoint")
        self.client.list_assignments.return_value = [SimpleNamespace(
            endpoint="external-endpoint", variant=AssignmentVariant.GPU,
            accelerator=Accelerator.L4, machine_shape=Shape.STANDARD,
        )]
        self.assertIsNone(self.execute("sessions"))
        self.assertIn("[?] external-endpoint | Hardware: L4", self.output.getvalue())
        self.assertEqual(self.store.get("shared").endpoint, "first-account-endpoint")

    def test_empty_local_store_does_not_swallow_authentication_exit(self):
        self.client.list_assignments.side_effect = SystemExit(1)
        self.assertEqual(self.execute("sessions"), 1)
        self.assertNotIn("No active sessions", self.output.getvalue())

    def test_bad_store_is_unknown_and_prevents_allocation(self):
        Path(self.store.path).write_text("broken JSON", encoding="utf-8")
        self.assertIn("failed", self.execute("new", "-s", "shared", "--gpu", "L4"))
        self.client.assign.assert_not_called()

    def test_new_receipt_uses_endpoint_written_by_that_command(self):
        self.client.assign.return_value = SimpleNamespace(endpoint="assigned-endpoint")
        self.assertIsNone(self.execute("new", "-s", "shared", "--gpu", "L4"))
        self.assertIn("CSTUDIO_ASSIGNED assigned-endpoint", self.output.getvalue())
        self.assertEqual(self.store.get("shared").endpoint, "assigned-endpoint")

    def test_colliding_alias_during_new_is_preserved_and_yields_no_receipt(self):
        def assign(*_args, **_kwargs):
            self.register("replacement")
            return SimpleNamespace(endpoint="new-endpoint")

        self.client.assign.side_effect = assign
        self.assertIn("failed", self.execute("new", "-s", "shared", "--gpu", "L4"))
        self.assertNotIn("CSTUDIO_ASSIGNED", self.output.getvalue())
        self.assertEqual(self.store.get("shared").endpoint, "replacement")

    def test_stop_refuses_replacement_before_dispatch(self):
        self.register("replacement")
        self.assertIn("failed", self.execute("stop", "-s", "shared", endpoint="original"))
        self.client.unassign.assert_not_called()
        self.runtime.stop.assert_not_called()

    def test_stop_preserves_alias_replaced_after_runtime_resolution(self):
        self.register("original")
        self.client.unassign.side_effect = lambda _endpoint: self.register("replacement")
        self.assertIsNone(self.execute("stop", "-s", "shared", endpoint="original"))
        self.client.unassign.assert_called_once_with("original")
        self.assertEqual(self.store.get("shared").endpoint, "replacement")

    def test_scoped_adapter_redacts_bare_oauth_and_proxy_tokens_before_writing_output(self):
        self.register("original")
        updated_proxy = "fixture-rotated-proxy-secret"
        access = "fixture-adapter-access-secret"
        self.token.write_text(json.dumps(dict(self.grant, token=access)), encoding="utf-8")

        def command(**_options):
            store = _colab.cli.state.store
            runtime = store.get("shared")
            updated = runtime.model_copy(update={"token": updated_proxy})
            store.add(updated)
            print(f"stdout: {access} {self.grant['refresh_token']} {runtime.token} {updated_proxy}")
            print(f"stderr: {self.grant['client_secret']}", file=sys.stderr)
            return 0

        with patch.object(_colab.cli, "app", side_effect=command), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(self.execute("download", "-s", "shared", endpoint="original"), 0)
        emitted = self.output.getvalue() + errors.getvalue()
        for private in (access, self.grant["refresh_token"], self.grant["client_secret"],
                        "fixture-proxy-token", updated_proxy):
            self.assertNotIn(private, emitted)
        self.assertEqual(emitted.count("[redacted]"), 5)
        self.assertEqual(self.store.get("shared").token, updated_proxy)
        if os.name == "posix":
            self.assertEqual(Path(self.store.path).stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
