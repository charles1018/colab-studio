"""Offline checks for the shared shutdown confirmation contract."""
import io
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from fastapi import HTTPException

from cstudio import cli
from cstudio import colab_client as C
from cstudio import webui


NO_SESSIONS = "[colab] No active sessions found on server."


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.keep = Path(self.enterContext(tempfile.TemporaryDirectory())) / "keep"
        self.runtimes: dict[str, str] = {}
        self.refuse_stop: set[str] = set()
        self.commands: list[tuple] = []
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.multiple(
            C,
            KEEP=self.keep,
            account_scope=Mock(return_value="fixture-account"),
            runtime_endpoint=Mock(side_effect=lambda name, **_: self.runtimes.get(name)),
            colab=Mock(side_effect=self.fake_command),
            start_watchdog=Mock(),
            token_ok=Mock(return_value=False),
        ))
        self.ui = webui.StudioUIState()
        self.enterContext(patch.object(webui, "STATE", self.ui))

    def fake_command(self, command, *args, **options):
        self.commands.append((command, *args))
        if command == "stop":
            self.assertTrue(options.get("check", True))
            if args[1] not in self.refuse_stop:
                self.runtimes.pop(args[1], None)
            return "[colab] Session terminated."
        if command == "sessions":
            self.assertTrue(options.get("check"))
            return "\n".join(
                f"[{name}] {endpoint} | Hardware: L4 | Shape: Standard | Variant: GPU"
                for name, endpoint in self.runtimes.items()
            ) or NO_SESSIONS
        self.fail(f"Unexpected Colab command {command}")

    def session(self, name, owner="owned"):
        self.runtimes[name] = name + "-endpoint"
        C.set_keep(name, 5_000, owner=owner, extra={
            "runtime_endpoint": self.runtimes[name], "account_scope": "fixture-account",
        })

    def test_success_reply_without_verified_release_remains_pending(self):
        self.session("gpu")
        self.refuse_stop.add("gpu")
        self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.PENDING)
        self.assertTrue(C.keep_info("gpu")["release_pending"])
        C.start_watchdog.assert_called_once_with("gpu")
        self.assertIn("gpu", self.runtimes)

    def test_already_scheduled_shutdown_checks_status_without_second_stop(self):
        self.session("gpu")
        self.refuse_stop.add("gpu")
        self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.PENDING)
        C.set_keep("gpu", C.time.time() + 600)
        self.assertLess(C.keep_info("gpu")["expires"], C.time.time() + 60)
        self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.PENDING)
        self.assertEqual(self.commands.count(("stop", "-s", "gpu")), 1)

    def test_external_explicit_stop_failure_does_not_schedule_retry(self):
        self.session("external", owner="external")
        self.refuse_stop.add("external")
        original = C.keep_info("external")
        self.assertEqual(C.release_runtime("external"), C.ReleaseState.FAILED)
        self.assertEqual(C.keep_info("external"), original)
        self.assertIn(("stop", "-s", "external"), self.commands)
        C.start_watchdog.assert_not_called()

    def test_watchdog_launch_failure_keeps_record_but_does_not_claim_retry(self):
        self.session("gpu")
        self.refuse_stop.add("gpu")
        C.start_watchdog.side_effect = OSError("fixture process creation failed")
        self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.FAILED)
        self.assertTrue(C.keep_info("gpu")["release_pending"])

    def test_automatic_request_rechecks_ownership_at_dispatch(self):
        self.session("external", owner="external")
        self.assertEqual(C.release_runtime("external", owned_only=True), C.ReleaseState.FAILED)
        C.colab.assert_not_called()

    def test_unnamed_original_runtime_prevents_confirmation(self):
        self.session("gpu")
        old = self.runtimes.pop("gpu")
        self.runtimes["?"] = old
        self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.PENDING)
        self.assertEqual(self.commands, [("sessions",)])
        self.assertEqual(C.keep_info("gpu")["runtime_endpoint"], old)

    def test_unknown_runtime_identity_cannot_schedule_future_name_based_stop(self):
        C.set_keep("gpu", 5_000, owner="owned")
        self.runtimes["?"] = "unclaimed-endpoint"
        self.assertEqual(C.release_runtime("gpu", owned_only=True), C.ReleaseState.FAILED)
        self.assertEqual(self.commands, [])
        C.start_watchdog.assert_not_called()
        self.assertNotIn("release_pending", C.keep_info("gpu"))

    def test_legacy_name_only_ownership_does_not_stop_a_replacement(self):
        C.set_keep("gpu", 5_000, owner="owned")
        self.runtimes["gpu"] = "unproven-runtime"
        original = C.keep_info("gpu")
        self.assertEqual(C.release_runtime("gpu", owned_only=True), C.ReleaseState.FAILED)
        self.assertNotIn(("stop", "-s", "gpu"), self.commands)
        self.assertEqual(C.keep_info("gpu"), original)

    def test_other_accounts_empty_listing_cannot_confirm_or_remove_keep(self):
        self.session("gpu")
        self.runtimes.clear()
        C.account_scope.return_value = "other-account"
        original = C.keep_info("gpu")
        self.assertEqual(C.release_runtime("gpu", owned_only=True), C.ReleaseState.FAILED)
        self.assertEqual(C.owned_sessions(), [])
        self.assertEqual(C.keep_info("gpu"), original)
        C.colab.assert_not_called()

    def test_account_switch_during_listing_prevents_stop_and_confirmation(self):
        self.session("gpu")
        original = C.keep_info("gpu")

        def switch(*_args, **_options):
            C.account_scope.return_value = "other-account"
            return NO_SESSIONS

        C.colab.side_effect = switch
        with patch.object(C.time, "sleep"):
            self.assertEqual(C.release_runtime("gpu", owned_only=True), C.ReleaseState.FAILED)
        self.assertEqual(C.keep_info("gpu"), original)
        self.assertTrue(all(call.args == ("sessions",) for call in C.colab.call_args_list))

    def test_listing_failure_keeps_release_pending_without_dispatch(self):
        self.session("gpu")
        with patch.object(C, "sessions", side_effect=OSError("connection unavailable")), patch.object(C.time, "sleep"):
            self.assertEqual(C.release_runtime("gpu", owned_only=True), C.ReleaseState.PENDING)
        self.assertTrue(C.keep_info("gpu")["release_pending"])
        C.colab.assert_not_called()

    def test_parallel_stop_does_not_dispatch_again_while_first_request_is_in_flight(self):
        self.session("gpu")
        dispatching = Event()
        complete = Event()
        command = C.colab.side_effect

        def block_stop(op, *args, **options):
            if op == "stop":
                dispatching.set()
                self.assertTrue(complete.wait(timeout=5))
            return command(op, *args, **options)

        C.colab.side_effect = block_stop
        with ThreadPoolExecutor(max_workers=1) as pool:
            started = pool.submit(C.release_runtime, "gpu")
            try:
                self.assertTrue(dispatching.wait(timeout=5))
                self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.PENDING)
            finally:
                complete.set()
            self.assertEqual(started.result(timeout=5), C.ReleaseState.CONFIRMED)
        self.assertEqual(self.commands.count(("stop", "-s", "gpu")), 1)

    def test_web_keep_refuses_lease_replaced_after_validation(self):
        self.session("gpu")
        keep_target = C.keep_target

        def renew(*args):
            target = keep_target(*args)
            C.set_keep("gpu", 8_000)
            return target

        with patch.object(C, "keep_target", side_effect=renew):
            with self.assertRaises(HTTPException) as response:
                webui.api_colab_keep({"session": "gpu", "minutes": 30})
        self.assertEqual(response.exception.status_code, 409)
        self.assertEqual(C.keep_info("gpu")["expires"], 8_000)

    def test_external_named_assignment_without_keep_can_be_stopped_explicitly(self):
        self.runtimes["external"] = "external-endpoint"
        self.assertEqual(C.release_runtime("external"), C.ReleaseState.CONFIRMED)
        self.assertIn(("stop", "-s", "external"), self.commands)
        C.start_watchdog.assert_not_called()

    def test_explicit_shutdown_detects_stale_name_without_touching_replacement(self):
        self.session("gpu")
        self.runtimes["gpu"] = "replacement"
        self.assertEqual(C.release_runtime("gpu"), C.ReleaseState.FAILED)
        self.assertEqual(self.commands, [("sessions",)])
        self.assertEqual(C.keep_info("gpu")["runtime_endpoint"], "gpu-endpoint")

    def test_cli_bulk_reports_partial_release_and_skips_unowned_targets(self):
        self.session("first")
        self.session("second")
        self.session("external", owner="external")
        self.runtimes["cstudio-foreign"] = "unowned-assignment"
        self.refuse_stop.add("second")

        with self.assertRaises(SystemExit) as interrupted:
            cli.cmd_stop(Namespace(all=True, session=None))
        self.assertEqual(interrupted.exception.code, 1)
        self.assertEqual([cmd[2] for cmd in self.commands if cmd[0] == "stop"], ["first", "second"])
        self.assertIn("已確認 first 關閉", self.stdout.getvalue())
        self.assertIn("second 尚待確認", self.stdout.getvalue())
        self.assertIsNone(C.keep_info("first"))
        self.assertTrue(C.keep_info("second")["release_pending"])

    def test_web_bulk_reports_partial_success_without_aborting_external_run(self):
        self.session("first")
        self.session("second")
        self.session("external", owner="external")
        self.ui.session = "external"
        self.ui.running = True
        self.refuse_stop.add("second")

        response = webui.api_colab_stop({"all": True})
        result = json.loads(response.body)
        self.assertEqual(response.status_code, 202)
        self.assertFalse(result["ok"])
        self.assertEqual(result["verified_gone"], ["first"])
        self.assertEqual(result["stopped"], ["first"])
        self.assertEqual(result["pending"], ["second"])
        self.assertIsNone(self.ui.abort_action)
        self.assertEqual(self.ui.session, "external")
        self.assertIn("external", self.runtimes)

    def test_web_explicit_external_shutdown_does_not_abort_a_different_batch(self):
        self.session("target", owner="external")
        self.ui.session = "different"
        self.ui.running = True
        response = webui.api_colab_stop({"session": "target"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["verified_gone"], ["target"])
        self.assertIsNone(self.ui.abort_action)
        C.start_watchdog.assert_not_called()

    def test_web_bulk_waits_for_its_own_inflight_allocation(self):
        self.ui.session = "new-studio"
        self.ui.running = True
        self.ui.provisioning = True
        response = webui.api_colab_stop({"all": True})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(response.body)["pending"], ["new-studio"])
        self.assertEqual(self.ui.abort_action, "stop")
        C.colab.assert_not_called()

    def test_web_rejects_truthy_non_boolean_bulk_values(self):
        self.session("gpu")
        for value in ["false", "true", 1, None, {}]:
            with self.subTest(value=value), self.assertRaises(HTTPException) as rejected:
                webui.api_colab_stop({"all": value})
            self.assertEqual(rejected.exception.status_code, 400)
        C.colab.assert_not_called()

    def test_web_reports_failed_stop_without_claiming_other_sessions(self):
        self.session("external", owner="external")
        self.ui.session = "external"
        self.refuse_stop.add("external")
        response = webui.api_colab_stop({"session": "external"})
        self.assertEqual(response.status_code, 502)
        result = json.loads(response.body)
        self.assertEqual(result["stopped"], [])
        self.assertEqual(result["failed"], ["external"])
        self.assertEqual(self.ui.session, "external")
        C.start_watchdog.assert_not_called()

    def test_web_status_clears_pending_only_after_confirmed_absence(self):
        self.session("gpu")
        self.ui.session = "gpu"
        self.refuse_stop.add("gpu")
        self.assertEqual(webui.api_colab_stop({"all": True}).status_code, 202)
        self.assertEqual(self.ui.stage, "shutdown_pending")
        with patch.object(C, "token_ok", return_value=True), patch.object(C, "usage", return_value=(None, "")):
            self.ui.refresh_colab_cache(force=True)
            self.assertEqual(self.ui.session, "gpu")
            self.runtimes.clear()
            self.runtimes["?"] = "unrelated-unassigned-alias"
            self.assertEqual(C.release_runtime("gpu", watchdog_keep=C.keep_info("gpu")), C.ReleaseState.CONFIRMED)
            C.account_scope.return_value = "other-account"
            self.ui.refresh_colab_cache(force=True)
            self.assertEqual(self.ui.stage, "shutdown_pending")
            C.account_scope.return_value = "fixture-account"
            self.ui.refresh_colab_cache(force=True)
        self.assertIsNone(self.ui.session)
        self.assertEqual(self.ui.stage, "stopped")


class AssignmentVerificationTests(unittest.TestCase):
    def test_missing_endpoint_is_verifiable_with_unrelated_unnamed_assignments(self):
        out = "[?] unrelated-endpoint | Hardware: L4 | Shape: Standard | Variant: GPU"
        self.assertTrue(C.session_gone("old-alias", [], out, "released-endpoint"))
        self.assertFalse(C.session_gone("old-alias", [], out, "unrelated-endpoint"))
        self.assertFalse(C.session_gone("old-alias", [], out))

    def test_empty_listing_with_authentication_warning_is_not_authoritative(self):
        output = NO_SESSIONS + "\nTo authorize colab-cli, visit this URL in any browser:"
        self.assertFalse(C.session_gone("gpu", [], output, "known-endpoint"))

    def test_benign_cli_messages_do_not_mask_verified_absence(self):
        output = (
            "[colab] A new version of Colab CLI is available: 0.8.0 (current: 0.7.4)\n"
            "[colab] Run 'colab update' to update.\n\n"
            "[colab] Pruned 1 stale local session(s).\n" + NO_SESSIONS
        )
        self.assertTrue(C.session_gone("gpu", [], output, "known-endpoint"))

    def test_nonzero_exit_code_invalidates_even_a_complete_looking_list(self):
        response = subprocess.CompletedProcess(["fixture"], 1, stdout=NO_SESSIONS, stderr="")
        with patch.object(C.subprocess, "run", return_value=response) as command, patch.object(C, "account_scope", return_value="fixture-account"):
            names, out = C.sessions_checked(tries=1)
        command.assert_called_once()
        self.assertIsNone(names)
        self.assertFalse(C.session_gone("gpu", names, out, "known-endpoint"))


if __name__ == "__main__":
    unittest.main()
