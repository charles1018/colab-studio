"""Offline coverage for resource recovery around a Colab allocation."""
import hashlib
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from cstudio import colab_client as C
from cstudio import runner as R


class RuntimeAllocationTests(unittest.TestCase):
    session = "cstudio-test"
    endpoint = "fixture-runtime"
    no_sessions = "[colab] No active sessions found on server."
    image_bytes = b"fixture image"

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.manifest = self.directory / "manifest.json"
        self.manifest.write_text(
            json.dumps({"jobs": [{"id": "test", "prompt": "a tree"}]}), encoding="utf-8"
        )
        self.result_path = self.directory / "out" / "results.json"
        self.now = 1_000.0
        self.runtimes: dict[str, str] = {}
        self.abort = None
        self.on_allocate = Mock()
        self.on_stop = Mock()
        self.reservations = []
        self.commands = []

        self.log = self.enterContext(redirect_stdout(io.StringIO()))
        clock = SimpleNamespace(time=lambda: self.now, sleep=self.advance, strftime=time.strftime)
        self.enterContext(patch.object(R, "time", clock))
        self.enterContext(patch.multiple(
            C,
            KEEP=self.directory / "keep",
            time=clock,
            colab=Mock(side_effect=self.fake_colab),
            token_ok=Mock(return_value=True),
            runtime_endpoint=Mock(side_effect=self.runtimes.get),
            start_watchdog=Mock(side_effect=self.start_fake_watchdog),
            poll_remote_status=Mock(return_value=self.done_status()),
            remote_exec=Mock(return_value='CSTUDIO_PREP {"launched": true, "missing_inputs": []}'),
        ))

    def advance(self, seconds):
        self.now += seconds

    def start_fake_watchdog(self, session):
        self.reservations.append(C.keep_info(session))

    def fake_colab(self, command, *args, **options):
        self.commands.append(command)
        if command == "usage":
            return "100 compute units"
        if command == "sessions":
            return "\n".join(
                f"[{name}] {endpoint} | Hardware: L4" for name, endpoint in self.runtimes.items()
            ) or self.no_sessions
        if command == "new":
            self.assertEqual(args, ("-s", self.session, "--gpu", "L4"))
            self.assertFalse(C.is_owned_session(self.session))
            self.assertEqual(self.reservations[0]["created_by"], "pending")
            self.assertGreater(self.reservations[0]["busy_until"], self.now)
            self.runtimes[self.session] = self.endpoint
            self.on_allocate()
            return "[colab] Session READY."
        if command == "stop":
            self.on_stop()
            self.runtimes.pop(args[1], None)
            return "[colab] Session terminated."
        if command == "download":
            Path(args[-1]).write_bytes(self.image_bytes)
            return ""
        raise AssertionError(f"Unexpected Colab operation: {command}")

    def done_status(self):
        return {
            "done": True,
            "stage": "done",
            "jobs": {"test": {
                "status": "done", "file": "out/test.png",
                "sha256": hashlib.sha256(self.image_bytes).hexdigest(),
            }},
        }

    def run_manifest(self, **options):
        return R.run_manifest(
            self.manifest, session=self.session, exit_on_error=False,
            should_abort=lambda: self.abort, **options,
        )

    def test_abort_before_preflight_does_not_allocate(self):
        self.abort = "stop"
        with self.assertRaisesRegex(RuntimeError, "stopped by user"):
            self.run_manifest()
        C.colab.assert_not_called()
        self.assertIsNone(C.keep_info(self.session))

    def test_cancel_after_reservation_before_allocation_removes_reservation(self):
        def arm_and_cancel(session):
            self.start_fake_watchdog(session)
            self.abort = "cancel"

        C.start_watchdog.side_effect = arm_and_cancel
        with self.assertRaisesRegex(RuntimeError, "cancelled by user"):
            self.run_manifest()
        self.assertNotIn("new", self.commands)
        self.assertNotIn("stop", self.commands)
        self.assertIsNone(C.keep_info(self.session))

    def test_cancel_during_allocation_transitions_to_finite_idle_lease(self):
        self.on_allocate.side_effect = lambda: setattr(self, "abort", "cancel")
        with self.assertRaisesRegex(RuntimeError, "cancelled by user"):
            self.run_manifest(stop_after=True)
        keep = C.keep_info(self.session)
        self.assertEqual(keep["created_by"], "owned")
        self.assertEqual(keep["runtime_endpoint"], self.endpoint)
        self.assertEqual(keep["expires"], self.now + 30 * 60)
        self.assertIsNone(keep["busy_until"])
        self.assertEqual(self.commands.count("new"), 1)
        self.assertNotIn("stop", self.commands)
        C.remote_exec.assert_not_called()

    def test_stop_during_allocation_waits_for_confirmation(self):
        self.on_allocate.side_effect = lambda: setattr(self, "abort", "stop")
        with self.assertRaisesRegex(RuntimeError, "stopped by user"):
            self.run_manifest()
        self.assertEqual(self.commands, ["usage", "sessions", "new", "stop", "sessions", "usage"])
        self.assertNotIn(self.session, self.runtimes)
        self.assertIsNone(C.keep_info(self.session))
        C.remote_exec.assert_not_called()

    def test_stop_transport_failure_leaves_watchdog_release_requested(self):
        self.on_allocate.side_effect = lambda: setattr(self, "abort", "stop")
        self.on_stop.side_effect = OSError("fixture transport failed")
        with self.assertRaisesRegex(RuntimeError, "stopped by user"):
            self.run_manifest()

        keep = C.keep_info(self.session)
        self.assertEqual(keep["created_by"], "owned")
        self.assertTrue(keep["release_pending"])
        self.assertEqual(keep["expires"], self.now + 30)
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        self.assertTrue(result["shutdown_pending"])

        self.on_stop.side_effect = None
        C.watchdog_loop(self.session)
        self.assertEqual(self.commands.count("stop"), 2)
        C.poll_remote_status.assert_not_called()
        self.assertIsNone(C.keep_info(self.session))

    def test_stop_with_lost_reply_still_verifies_release(self):
        self.on_allocate.side_effect = lambda: setattr(self, "abort", "stop")

        def stop_and_lose_reply():
            self.runtimes.pop(self.session)
            raise OSError("fixture stop reply lost")

        self.on_stop.side_effect = stop_and_lose_reply
        with self.assertRaisesRegex(RuntimeError, "stopped by user"):
            self.run_manifest()
        self.assertIsNone(C.keep_info(self.session))
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        self.assertNotIn("shutdown_pending", result)

    def test_allocation_transport_failure_leaves_unclaimed_reservation(self):
        self.on_allocate.side_effect = RuntimeError("fixture allocation reply lost")
        with self.assertRaisesRegex(RuntimeError, "reply lost"):
            self.run_manifest()
        keep = C.keep_info(self.session)
        self.assertEqual(keep["created_by"], "pending")
        self.assertEqual(keep["busy_until"], self.now + 30)
        self.assertFalse(C.is_owned_session(self.session))

        with self.assertRaisesRegex(RuntimeError, "仍待確認"):
            self.run_manifest()
        self.assertEqual(self.commands.count("new"), 1)
        self.assertNotIn("stop", self.commands)
        C.remote_exec.assert_not_called()

    def test_missing_endpoint_after_allocation_keeps_state_unconfirmed(self):
        self.on_allocate.side_effect = self.runtimes.clear
        with self.assertRaisesRegex(RuntimeError, "遠端識別"):
            self.run_manifest()
        self.assertEqual(C.keep_info(self.session)["created_by"], "pending")
        self.assertNotIn("stop", self.commands)
        C.remote_exec.assert_not_called()

    def test_watchdog_arm_failure_prevents_allocation(self):
        C.start_watchdog.side_effect = OSError("fixture launch failed")
        with self.assertRaisesRegex(OSError, "launch failed"):
            self.run_manifest()
        self.assertNotIn("new", self.commands)
        self.assertIsNone(C.keep_info(self.session))

    def test_keep_write_failure_after_allocation_releases_known_runtime(self):
        write = C.set_keep

        def fail_promotion(session, expires, **opts):
            if opts.get("owner") == "owned" and expires is None:
                raise OSError("fixture write failed")
            write(session, expires, **opts)

        with patch.object(C, "set_keep", side_effect=fail_promotion):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.run_manifest()
        self.assertEqual(self.commands.count("new"), 1)
        self.assertEqual(self.commands.count("stop"), 1)
        self.assertIsNone(C.keep_info(self.session))

    def test_cancel_without_a_writable_idle_lease_releases_new_runtime(self):
        write = C.set_keep
        self.on_allocate.side_effect = lambda: setattr(self, "abort", "cancel")

        def fail_idle_write(session, expires, **opts):
            if expires == self.now + 30 * 60:
                raise OSError("fixture idle write failed")
            write(session, expires, **opts)

        with patch.object(C, "set_keep", side_effect=fail_idle_write):
            with self.assertRaisesRegex(OSError, "idle write failed"):
                self.run_manifest()
        self.assertEqual(self.commands.count("stop"), 1)
        self.assertIsNone(C.keep_info(self.session))

    def test_unavailable_listing_prevents_allocation(self):
        with patch.object(C, "sessions", return_value=([], "")):
            with self.assertRaisesRegex(RuntimeError, "無法確認開啟中的"):
                self.run_manifest()
        self.assertNotIn("new", self.commands)
        self.assertIsNone(C.keep_info(self.session))

    def test_successful_allocation_syncs_and_transitions_to_idle(self):
        result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "done")
        self.assertEqual((self.result_path.parent / "test.png").read_bytes(), self.image_bytes)
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 15 * 60)
        self.assertIsNone(C.keep_info(self.session)["busy_until"])
        self.assertEqual(self.commands.count("new"), 1)
        self.assertNotIn("stop", self.commands)

    def test_reuse_owned_runtime_keeps_its_bundle_cache(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now + 60, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "bundle_hash": R.compute_remote_bundle_hash(),
        })
        C.poll_remote_status.side_effect = [{"alive": False}, self.done_status()]
        with patch.object(R, "bundle_and_launch_code", wraps=R.bundle_and_launch_code) as bundle:
            result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "done")
        self.assertIsNone(bundle.call_args.args[0])
        self.assertEqual(C.keep_info(self.session)["runtime_endpoint"], self.endpoint)
        self.assertNotIn("new", self.commands)
        self.assertNotIn("stop", self.commands)

    def test_reuse_external_runtime_never_arms_a_hard_deadline(self):
        self.runtimes[self.session] = self.endpoint
        C.poll_remote_status.side_effect = [{"alive": False}, self.done_status()]
        self.assertEqual(self.run_manifest()["jobs"]["test"]["status"], "done")
        keep = C.keep_info(self.session)
        self.assertEqual(keep["created_by"], "external")
        self.assertIsNone(keep["busy_until"])
        self.advance(15 * 60)
        C.watchdog_loop(self.session)
        self.assertIn(self.session, self.runtimes)
        self.assertNotIn("new", self.commands)
        self.assertNotIn("stop", self.commands)

    def test_watchdog_releases_confirmed_owned_runtime_at_busy_deadline(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, None, owner="owned", extra={
            "busy_until": self.now + 30, "runtime_endpoint": self.endpoint,
        })
        C.watchdog_loop(self.session)
        self.assertNotIn(self.session, self.runtimes)
        self.assertEqual(self.commands, ["stop", "sessions"])
        C.poll_remote_status.assert_not_called()
        self.assertIsNone(C.keep_info(self.session))

    def test_watchdog_preserves_unconfirmed_allocation_until_verified_absent(self):
        C.set_keep(self.session, None, owner="pending", extra={"busy_until": self.now})
        listings = iter([([], "[?] fixture-runtime | Hardware: L4"), ([], self.no_sessions)])

        def list_while_pending():
            self.assertEqual(C.keep_info(self.session)["created_by"], "pending")
            return next(listings)

        with patch.object(C, "sessions_checked", side_effect=list_while_pending) as checks:
            C.watchdog_loop(self.session)
        self.assertEqual(checks.call_count, 2)
        self.assertNotIn("stop", self.commands)
        self.assertIsNone(C.keep_info(self.session))

    def test_release_does_not_stop_a_different_runtime_with_same_name(self):
        self.runtimes[self.session] = "replacement-runtime"
        self.assertFalse(C.release_runtime(self.session, "owned", self.endpoint))
        self.assertEqual(self.commands, ["sessions"])
        self.assertEqual(C.keep_info(self.session)["runtime_endpoint"], self.endpoint)

    def test_cancel_during_retry_backoff_prevents_second_allocation(self):
        abort = Mock(side_effect=[None, None, RuntimeError("fixture cancelled")])
        with patch.object(C, "colab", side_effect=RuntimeError("503")) as command:
            with self.assertRaisesRegex(RuntimeError, "fixture cancelled"):
                C.new_runtime(self.session, "L4", retries=2, wait=3, check_abort=abort)
        command.assert_called_once()
        self.assertEqual(self.now, 1_003.0)

    def test_allocation_deadline_does_not_shorten_in_flight_request(self):
        with patch.object(C, "colab", side_effect=RuntimeError("503")) as command:
            with self.assertRaisesRegex(RuntimeError, "配置時間上限"):
                C.new_runtime(self.session, "L4", wait=120, deadline=self.now + 12)
        self.assertEqual(command.call_args_list, [
            call("new", "-s", self.session, "--gpu", "L4", timeout=C.ASSIGN_TIMEOUT),
        ])
        self.assertEqual(self.now, 1_012.0)

    def test_unconfirmed_timeout_never_retries_the_allocation(self):
        # The timeout text may contain the Session's name and numerical status substrings.
        with patch.object(C, "colab", side_effect=C.ColabTimeout("fixture-503 timed out")) as command:
            with self.assertRaises(C.ColabTimeout):
                C.new_runtime(self.session, "L4", retries=2)
        command.assert_called_once()

    def test_retry_does_not_overwrite_a_registered_assignment(self):
        self.on_allocate.side_effect = RuntimeError("503")
        with self.assertRaisesRegex(RuntimeError, "503"):
            self.run_manifest()
        self.assertEqual(self.commands.count("new"), 1)
        self.assertEqual(C.keep_info(self.session)["created_by"], "pending")

    def test_allocation_returning_after_deadline_is_released(self):
        self.on_allocate.side_effect = lambda: self.advance(61)
        with self.assertRaisesRegex(RuntimeError, "分鐘上限"):
            self.run_manifest(max_minutes=1)
        self.assertEqual(self.commands.count("new"), 1)
        self.assertEqual(self.commands.count("stop"), 1)
        C.remote_exec.assert_not_called()
        self.assertIsNone(C.keep_info(self.session))

    def test_cancel_after_launch_waits_for_confirmation_and_syncs_completed_jobs(self):
        progress = Mock()

        def prepare(_session, _code, **_opts):
            self.abort = "cancel"
            return 'CSTUDIO_PREP {"launched": true, "missing_inputs": []}'

        def snapshot(_session):
            run_id = progress.call_args.args[0]["run_id"]
            if C.poll_remote_status.call_count == 1:
                return {"run_id": run_id, "stage": "running", "done": False}
            return dict(self.done_status(), run_id=run_id, stage="cancelled", cancel_protocol=1)

        C.remote_exec.side_effect = prepare
        C.poll_remote_status.side_effect = snapshot
        with patch.object(C, "request_cancel", return_value={"state": "pending"}) as send:
            with self.assertRaisesRegex(R.RunCancelled, "已確認遠端批次取消"):
                self.run_manifest(progress_cb=progress)
        cancelled = progress.call_args.args[0]
        send.assert_called_once_with(self.session, cancelled["run_id"], allow_start=True)
        self.assertEqual(cancelled["stage"], "cancelled")
        self.assertEqual(cancelled["_local_synced_jobs"], ["test"])
        self.assertEqual((self.result_path.parent / "test.png").read_bytes(), self.image_bytes)
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 30 * 60)
        self.assertNotIn("stop", self.commands)

    def test_failed_remote_cancel_preserves_existing_recovery_deadline(self):
        progress = Mock()

        def prepare(_session, _code, **_opts):
            self.abort = "cancel"
            return 'CSTUDIO_PREP {"launched": true, "missing_inputs": []}'

        C.remote_exec.side_effect = prepare
        C.poll_remote_status.side_effect = lambda _session: {
            "run_id": progress.call_args.args[0]["run_id"], "stage": "cancel_failed",
            "done": True, "error": "fixture: remote task still active", "cancel_protocol": 1,
        }
        with patch.object(C, "request_cancel", return_value={"state": "pending"}):
            result = self.run_manifest(progress_cb=progress)
        self.assertEqual(result["error"], "fixture: remote task still active")
        self.assertIsNone(C.keep_info(self.session)["expires"])
        self.assertEqual(C.keep_info(self.session)["busy_until"], 1_000.0 + 120 * 60)

    def test_an_unconfirmed_cancellation_blocks_reuse(self):
        self.runtimes[self.session] = self.endpoint
        C.poll_remote_status.return_value = {"stage": "cancel_failed", "alive": False}
        with self.assertRaisesRegex(RuntimeError, "待確認的取消"):
            self.run_manifest()
        C.remote_exec.assert_not_called()
        self.assertNotIn("new", self.commands)
        self.assertNotIn("stop", self.commands)

    def test_lost_cancel_reply_retries_without_treating_it_as_confirmation(self):
        progress = Mock()
        C.remote_exec.side_effect = lambda *_args, **_opts: (
            setattr(self, "abort", "cancel") or 'CSTUDIO_PREP {"launched": true, "missing_inputs": []}'
        )

        def snapshot(_session):
            run_id = progress.call_args.args[0]["run_id"]
            if C.poll_remote_status.call_count == 1:
                return {"run_id": run_id, "stage": "running", "done": False}
            return dict(self.done_status(), run_id=run_id)

        C.poll_remote_status.side_effect = snapshot
        with patch.object(C, "request_cancel", side_effect=[RuntimeError("lost reply"), {"state": "finished"}]) as send:
            result = self.run_manifest(progress_cb=progress)
        self.assertEqual(send.call_count, 2)
        self.assertNotIn("error", result)
        self.assertEqual(progress.call_args.args[0]["stage"], "done")
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 15 * 60)


if __name__ == "__main__":
    unittest.main()
