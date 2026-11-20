"""Offline coverage for resource recovery around a Colab allocation."""
import hashlib
import io
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from PIL import Image

from cstudio import colab_client as C
from cstudio import results as store
from cstudio import runner as R


def _png(color):
    with io.BytesIO() as image:
        Image.new("RGB", (4, 4), color).save(image, format="PNG")
        return image.getvalue()


def _workspace_worker(manifest, run_id, pause_at, failed_jobs, ready, resume, reply):
    fixture = RuntimeAllocationTests()
    try:
        fixture.setUp()
        fixture.manifest = manifest
        fixture.image_bytes = _png("red")
        template = fixture.done_status()
        receipt = template["jobs"]["test"]
        jobs = json.loads(manifest.read_text())["jobs"]
        template["jobs"] = {
            jid: ({"status": "error", "error": f"{jid} fixture failure"} if jid in failed_jobs
                  else dict(receipt, file=f"out/{jid}.png"))
            for jid in (job["id"] for job in jobs)
        }
        fixture.status.return_value = template

        def pause():
            ready.set()
            if not resume.wait(10):
                raise RuntimeError("fixture did not resume")

        def colab(command, *args, **options):
            if command == "download" and pause_at == "download" and not ready.is_set():
                pause()
            if command == "usage" and pause_at == "finish" and "usage" in fixture.commands:
                pause()
            return fixture.fake_colab(command, *args, **options)

        C.colab.side_effect = colab
        if pause_at == "publish":
            save_results = store.save_results

            def pause_publication(path, data):
                if path.name == "results.json" and not ready.is_set():
                    pause()
                save_results(path, data)

            fixture.enterContext(patch.object(store, "save_results", side_effect=pause_publication))
        if pause_at == "allocation":
            def fail_allocation(*_, **__):
                pause()
                raise OSError("fixture allocation failed")

            fixture.enterContext(patch.object(C, "new_runtime", side_effect=fail_allocation))
        if run_id:
            fixture.runtimes[fixture.session] = fixture.endpoint
            result = R.recover_outputs(manifest, run_id)
        else:
            result = fixture.run_manifest()
        reply.send(("ok", result))
    except BaseException as exc:
        reply.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        fixture.doCleanups()
        reply.close()


class RuntimeAllocationTests(unittest.TestCase):
    session = "cstudio-test"
    endpoint = "fixture-runtime"
    no_sessions = "[colab] No active sessions found on server."
    allocation_refused = "[colab] Allocation refused (precondition failed)."
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
        self.uploaded = {}
        self.real_colab = C.colab
        self.status = Mock(return_value=self.done_status())
        self.bundle_hash = R.compute_remote_bundle_hash()

        self.log = self.enterContext(redirect_stdout(io.StringIO()))
        self.clock = SimpleNamespace(time=lambda: self.now, monotonic=lambda: self.now, sleep=self.advance, strftime=time.strftime)
        self.enterContext(patch.object(R, "time", self.clock))
        self.enterContext(patch.multiple(
            C,
            KEEP=self.directory / "keep",
            account_scope=Mock(return_value="fixture-account"),
            time=self.clock,
            colab=Mock(side_effect=self.fake_colab),
            token_ok=Mock(return_value=True),
            runtime_endpoint=Mock(side_effect=lambda name, **_: self.runtimes.get(name)),
            start_watchdog=Mock(side_effect=self.start_fake_watchdog),
            poll_remote_status=Mock(side_effect=self.poll_status),
            remote_exec=Mock(side_effect=lambda *_args, **_opts: self.prep_reply()),
        ))

    def poll_status(self, session, *, run_id=None, **opts):
        st = self.status(session, **opts)
        if run_id is None or not isinstance(st, dict):
            return st
        return {"bundle_hash": self.bundle_hash, "run_id": run_id, **st}

    def prep_reply(self, *, state="running", **extra):
        run_id = C.keep_info(self.session)["active_run_id"]
        return "CSTUDIO_PREP " + json.dumps({
            "state": state, "run_id": run_id, "bundle_hash": self.bundle_hash, "missing_inputs": [], **extra,
        })

    def advance(self, seconds):
        self.now += seconds

    def start_fake_watchdog(self, session, **_options):
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
            return f"CSTUDIO_ASSIGNED {self.endpoint}" if self.session in self.runtimes else "[colab] Session READY."
        if command == "stop":
            self.on_stop()
            self.runtimes.pop(args[1], None)
            return "[colab] Session terminated."
        if command == "download":
            Path(args[-1]).write_bytes(self.image_bytes)
            return ""
        if command == "upload":
            self.uploaded[args[-1]] = Path(args[-2]).read_bytes()
            return ""
        raise AssertionError(f"Unexpected Colab operation: {command}")

    def done_status(self):
        return {
            "done": True,
            "alive": False,
            "stage": "done",
            "jobs": {"test": {
                "status": "done", "file": "out/test.png",
                "bytes": len(self.image_bytes),
                "sha256": hashlib.sha256(self.image_bytes).hexdigest(),
                "execution": {
                    "pin_versions": True, "pin_models": True,
                    "environment": {"python": ["/fixture/python", "3.12"],
                                    "heads": {"ComfyUI": "fixture-commit"}, "packages": {"torch": "2.7"}},
                    "gpu": {"name": "fixture"},
                    "comfy": {"args": ["main.py"], "verified": True,
                              "versions": {"comfyui_version": "1", "python_version": "3.12", "pytorch_version": "2.7"}},
                    "models": [{"repo": "fixture", "file": "model.gguf", "path": "models/unet/model.gguf",
                                "bytes": 500, "sha256": "1" * 64, "revision": "2" * 40}],
                    "fitted_models": [],
                },
            }},
        }

    def run_manifest(self, **options):
        return R.run_manifest(
            self.manifest, session=self.session, exit_on_error=False,
            should_abort=lambda: self.abort, **options,
        )

    def _workspace_manifest(self, filename, *job_ids):
        path = self.directory / filename
        path.write_text(json.dumps({"jobs": [{"id": jid, "prompt": jid} for jid in job_ids]}))
        return path

    def _paused_workspace_worker(self, manifest, *, pause_at, run_id=None, failed_jobs=()):
        context = multiprocessing.get_context("spawn")
        ready, resume = context.Event(), context.Event()
        receive, reply = context.Pipe(duplex=False)
        process = context.Process(
            target=_workspace_worker,
            args=(manifest, run_id, pause_at, failed_jobs, ready, resume, reply),
        )
        process.start()
        reply.close()

        def cleanup():
            resume.set()
            if process.is_alive():
                process.terminate()
            process.join(5)
            receive.close()
            process.close()

        self.addCleanup(cleanup)
        self.assertTrue(ready.wait(10), "worker did not reach its pause")

        def finish():
            resume.set()
            self.assertTrue(receive.poll(10), "worker did not report")
            result = receive.recv()
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            return result

        return finish

    def test_abort_before_preflight_does_not_allocate(self):
        self.abort = "stop"
        with self.assertRaisesRegex(RuntimeError, "stopped by user"):
            self.run_manifest()
        C.colab.assert_not_called()
        self.assertIsNone(C.keep_info(self.session))

    def test_cancel_after_reservation_before_allocation_removes_reservation(self):
        def arm_and_cancel(session, **_options):
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
        self.assertEqual(self.commands, ["usage", "sessions", "new", "sessions", "stop", "sessions", "usage"])
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
            return write(session, expires, **opts)

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
            return write(session, expires, **opts)

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

    def test_mixed_error_and_assignment_rows_do_not_authorize_allocation(self):
        raw = "[other] other-endpoint | Hardware: L4\nPermission denied"
        with patch.object(C, "sessions", return_value=(["other"], raw)):
            with self.assertRaisesRegex(RuntimeError, "無法確認開啟中的"):
                self.run_manifest()
        self.assertNotIn("new", self.commands)

    def test_absent_alias_with_existing_keep_does_not_reallocate(self):
        C.set_keep(self.session, self.now + 60, owner="owned", extra={
            "runtime_endpoint": "previous-endpoint", "account_scope": "another-account",
        })
        original = C.keep_info(self.session)
        with self.assertRaises(RuntimeError):
            self.run_manifest()
        self.assertNotIn("new", self.commands)
        self.assertEqual(C.keep_info(self.session), original)

    def test_503_after_unrecorded_assignment_is_not_retried(self):
        def lose_reply():
            self.runtimes["?"] = self.runtimes.pop(self.session)
            raise RuntimeError("503 Service Unavailable")

        self.on_allocate.side_effect = lose_reply
        with self.assertRaises(RuntimeError):
            self.run_manifest(assign_retries=2, assign_wait=0)
        self.assertEqual(self.commands.count("new"), 1)
        self.assertNotIn("stop", self.commands)
        self.assertEqual(C.keep_info(self.session)["created_by"], "pending")

    def test_selected_alias_must_exist_for_web_reuse(self):
        with self.assertRaisesRegex(RuntimeError, "可沿用或安全配置"):
            self.run_manifest(reuse_only=True)
        self.assertNotIn("new", self.commands)

    def test_same_name_replacement_does_not_enter_the_runtime(self):
        self.runtimes[self.session] = "replacement-endpoint"
        C.set_keep(self.session, self.now + 60, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        original = C.keep_info(self.session)
        with self.assertRaisesRegex(RuntimeError, "遠端識別已變更"):
            self.run_manifest()
        C.remote_exec.assert_not_called()
        C.poll_remote_status.assert_not_called()
        self.assertNotIn("new", self.commands)
        self.assertEqual(C.keep_info(self.session), original)

    def test_web_selected_endpoint_rejects_replacement_even_without_a_local_keep(self):
        self.runtimes[self.session] = "replacement-endpoint"
        with self.assertRaisesRegex(RuntimeError, "Session 已變更"):
            self.run_manifest(reuse_only=True, expected_endpoint=self.endpoint)
        C.remote_exec.assert_not_called()
        C.poll_remote_status.assert_not_called()
        self.assertNotIn("new", self.commands)
        self.assertIsNone(C.keep_info(self.session))

    def test_reuse_progress_reports_assignment_hardware_separately_from_request(self):
        self.runtimes[self.session] = self.endpoint
        self.status.side_effect = [
            {"alive": False}, dict(self.done_status(), gpu_info={"name": "NVIDIA L4"}),
        ]
        progress = Mock()
        result = self.run_manifest(gpu="H100", reuse_only=True, expected_endpoint=self.endpoint, progress_cb=progress)
        events = [call.args[0] for call in progress.call_args_list]
        reused = next(event for event in events if event["stage"] == "reusing_vm")
        self.assertEqual(reused["_listed_hardware"], "L4")
        self.assertEqual(reused["_runtime_endpoint"], self.endpoint)
        synced = next(event for event in events if event["stage"] == "syncing_outputs")
        self.assertTrue(synced["_runtime_attached"])
        self.assertEqual(synced["gpu_info"]["name"], "NVIDIA L4")
        self.assertEqual(result["gpu"], "H100")
        self.assertNotIn("new", self.commands)

    def test_account_change_during_allocation_preserves_original_recovery_record(self):
        self.on_allocate.side_effect = lambda: setattr(C.account_scope, "return_value", "another-account")
        with self.assertRaisesRegex(RuntimeError, "登入身分已變更"):
            self.run_manifest()
        self.assertNotIn("stop", self.commands)
        self.assertEqual(C.keep_info(self.session)["account_scope"], "fixture-account")
        self.assertEqual(C.keep_info(self.session)["runtime_endpoint"], self.endpoint)
        C.remote_exec.assert_not_called()

    def test_auto_reuse_uses_current_account_owned_assignment(self):
        self.runtimes.update({"a-foreign": "other-endpoint", self.session: self.endpoint})
        for name, account in [("a-foreign", "other-account"), (self.session, "fixture-account")]:
            C.set_keep(name, self.now + 60, owner="owned", extra={
                "runtime_endpoint": self.runtimes[name], "account_scope": account,
            })
        self.status.side_effect = [{"alive": False}, self.done_status()]
        result = R.run_manifest(self.manifest, auto_reuse=True, exit_on_error=False)
        self.assertEqual(result["session"], self.session)
        self.assertNotIn("new", self.commands)

    def test_pending_allocation_blocks_web_auto_create_even_when_listing_is_empty(self):
        C.set_keep(self.session, None, owner="pending", extra={"account_scope": "fixture-account"})
        with self.assertRaisesRegex(RuntimeError, "尚待確認"):
            R.run_manifest(self.manifest, auto_reuse=True, exit_on_error=False)
        self.assertNotIn("new", self.commands)

    def test_name_only_legacy_keep_reuses_as_external_without_cache(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now + 60, owner="owned", extra={"bundle_hash": R.compute_remote_bundle_hash()})
        self.status.side_effect = [{"alive": False}, self.done_status()]
        with patch.object(R, "bundle_and_launch_code", wraps=R.bundle_and_launch_code) as bundle:
            self.run_manifest()
        self.assertIsNotNone(bundle.call_args.args[0])
        self.assertEqual(C.keep_info(self.session)["created_by"], "external")
        self.assertFalse(C.is_owned_session(self.session))

    def test_contended_allocation_reservation_does_not_replace_other_run(self):
        C.set_keep(self.session, None, owner="pending", extra={"allocation_id": "first"}, reserve=True)
        with self.assertRaises(FileExistsError):
            C.set_keep(self.session, None, owner="pending", extra={"allocation_id": "second"}, reserve=True)
        self.assertEqual(C.keep_info(self.session)["allocation_id"], "first")

    def test_successful_allocation_syncs_and_transitions_to_idle(self):
        result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "done")
        self.assertEqual((self.result_path.parent / "test.png").read_bytes(), self.image_bytes)
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 15 * 60)
        self.assertIsNone(C.keep_info(self.session)["busy_until"])
        self.assertEqual(self.commands.count("new"), 1)
        self.assertNotIn("stop", self.commands)

    def test_transport_and_remote_diagnostics_are_redacted_in_results_progress_and_console(self):
        status = self.done_status()
        status["log"] = ["refresh_token=fixture-log-refresh"]
        status["jobs"]["test"]["traceback"] = "code_verifier=fixture-job-verifier"
        self.status.return_value = status
        transport = C.colab.side_effect

        def fail_download(operation, *args, **options):
            if operation == "download":
                raise RuntimeError("HTTP download failed at https://example.test/asset?token=fixture-download-secret")
            return transport(operation, *args, **options)

        C.colab.side_effect = fail_download
        progress = Mock()
        result = self.run_manifest(progress_cb=progress)
        self.assertEqual(result["jobs"]["test"]["status"], "sync_error")
        for output in (self.result_path.read_text(encoding="utf-8"), self.log.getvalue(), str(progress.call_args_list)):
            self.assertIn("[redacted]", output)
            for secret in ("fixture-log-refresh", "fixture-job-verifier", "fixture-download-secret"):
                self.assertNotIn(secret, output)

    def edit_reference(self):
        ref = self.directory / "inputs" / "style" / "ref.png"
        ref.parent.mkdir(parents=True)
        ref.write_bytes(b"fixture")
        self.manifest.write_text(json.dumps({
            "jobs": [{"id": "test", "mode": "edit", "prompt": "fixture", "inputs": ["inputs/style/ref.png"]}],
        }), encoding="utf-8")
        return ref

    def test_upload_is_limited_to_the_declared_nested_input(self):
        ref = self.edit_reference()
        self.on_allocate.side_effect = lambda: ref.write_bytes(b"changed after staging")
        C.remote_exec.side_effect = lambda *_args, **_opts: (
            self.prep_reply(state="prepared", missing_inputs=["inputs/style/ref.png"])
            if C.remote_exec.call_count == 1 else self.prep_reply()
        )
        result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "done")
        args, = [call.args for call in C.colab.call_args_list if call.args[0] == "upload"]
        run_dir = f"/content/cstudio/runs/{result['run_id']}"
        remote_ref = f"{run_dir}/uploads/inputs/style/ref.png"
        self.assertEqual(args[:3], ("upload", "-s", self.session))
        self.assertEqual(args[-1], remote_ref)
        self.assertNotEqual(args[-2], ref)
        self.assertFalse(args[-2].exists())
        self.assertEqual(self.uploaded, {remote_ref: b"fixture"})
        self.assertEqual(ref.read_bytes(), b"changed after staging")
        download, = [call.args for call in C.colab.call_args_list if call.args[0] == "download"]
        self.assertEqual(download[-2], f"{run_dir}/out/test.png")

    def test_reference_modified_before_snapshot_fails_before_allocating(self):
        ref = self.edit_reference()
        reference_hashes = R._reference_hashes

        def hash_then_edit(*args):
            hashes = reference_hashes(*args)
            ref.write_bytes(b"changed before snapshot")
            return hashes

        with patch.object(R, "_reference_hashes", side_effect=hash_then_edit):
            with self.assertRaisesRegex(RuntimeError, "參考圖在批次準備期間已變更"):
                self.run_manifest()
        self.assertNotIn("new", self.commands)
        self.assertNotIn("upload", self.commands)

    def test_unexpected_remote_upload_path_stops_before_upload_or_worker_launch(self):
        self.edit_reference()
        C.remote_exec.side_effect = lambda *_args, **_opts: self.prep_reply(state="prepared", missing_inputs=["unlisted.png"])
        with self.assertRaisesRegex(RuntimeError, "待上傳清單"):
            self.run_manifest()
        C.remote_exec.assert_called_once()
        self.assertNotIn("upload", self.commands)

    def test_rejected_start_receipt_cannot_poll_or_download_other_run(self):
        self.edit_reference()
        C.remote_exec.side_effect = lambda *_args, **_opts: (
            self.prep_reply(state="prepared", missing_inputs=["inputs/style/ref.png"])
            if C.remote_exec.call_count == 1 else self.prep_reply(run_id="a" * 12)
        )
        with self.assertRaisesRegex(RuntimeError, "批次 ID 不符"):
            self.run_manifest()
        C.poll_remote_status.assert_not_called()
        self.assertNotIn("download", self.commands)

    def test_busy_receipt_protects_another_run_even_with_stop_after(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now + 60, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        self.status.return_value = {"alive": False}
        C.remote_exec.side_effect = lambda *_args, **_opts: self.prep_reply(state="busy", reason="busy", bundle_hash=None)
        with self.assertRaisesRegex(RuntimeError, "拒絕本批啟動"):
            self.run_manifest(stop_after=True)
        self.assertNotIn("stop", self.commands)
        self.assertIsNone(C.keep_info(self.session)["active_run_id"])
        self.assertEqual(C.poll_remote_status.call_count, 1)

    def test_old_status_or_mismatched_bundle_cannot_supply_results(self):
        self.status.side_effect = [
            dict(self.done_status(), run_id="a" * 12),
            dict(self.done_status(), bundle_hash="0" * 64),
            self.done_status(),
        ]
        result = self.run_manifest()
        download, = [call.args for call in C.colab.call_args_list if call.args[0] == "download"]
        self.assertEqual(download[-2], f"/content/cstudio/runs/{result['run_id']}/out/test.png")
        self.assertTrue(all(call.kwargs["run_id"] == result["run_id"] for call in C.poll_remote_status.call_args_list))
        self.assertEqual(C.poll_remote_status.call_count, 3)

    def test_stale_keep_bundle_hint_is_resubmitted_and_verified(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now + 60, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account", "bundle_hash": self.bundle_hash,
        })
        self.status.side_effect = [{"alive": False}, self.done_status()]
        C.remote_exec.side_effect = lambda *_args, **_opts: (
            self.prep_reply(state="bundle_required", bundle_hash=None) if C.remote_exec.call_count == 1 else self.prep_reply()
        )
        with patch.object(R, "bundle_and_launch_code", wraps=R.bundle_and_launch_code) as bundle:
            self.assertEqual(self.run_manifest()["jobs"]["test"]["status"], "done")
        self.assertIsNone(bundle.call_args_list[0].args[0])
        sent = bundle.call_args_list[1].args[0]
        self.assertEqual(R.bundle_digest(sent), self.bundle_hash)

    def test_symlinked_output_root_stops_before_any_colab_command(self):
        other_job = self.directory / "other"
        other_job.mkdir()
        try:
            self.result_path.parent.symlink_to(other_job, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links unavailable")
        with self.assertRaisesRegex(RuntimeError, "符號連結"):
            self.run_manifest()
        C.colab.assert_not_called()
        self.assertEqual(list(other_job.iterdir()), [])

    def test_unexpected_remote_result_filename_is_never_downloaded(self):
        status = self.done_status()
        status["jobs"]["test"]["file"] = "out/other.png"
        self.status.return_value = status
        result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "sync_error")
        self.assertNotIn("download", self.commands)
        self.assertFalse((self.result_path.parent / "test.png").exists())

    def test_disconnect_recovers_only_completed_jobs_from_the_original_run(self):
        self.image_bytes = _png("blue")
        self.manifest.write_text(json.dumps({"jobs": [
            {"id": jid, "prompt": jid} for jid in ("first", "second", "third")
        ]}), encoding="utf-8")
        first = dict(self.done_status()["jobs"]["test"], file="out/first.png")
        second = dict(first, file="out/second.png")
        self.status.side_effect = [
            {"done": False, "jobs": {"first": first}},
            OSError("disconnected"),
            {"done": False, "jobs": {"first": first, "second": second, "unexpected": second}},
            {"done": True, "jobs": {"first": first, "second": second}},
        ]
        with self.assertRaisesRegex(OSError, "disconnected"):
            self.run_manifest()
        old = json.loads(self.result_path.read_text())
        run_id = old["run_id"]
        self.assertEqual(set(old["jobs"]), {"first"})
        self.assertEqual(C.keep_info(self.session)["busy_until"], 8_200)
        self.assertNotIn("stop", self.commands)
        self.assertIn(run_id, self.log.getvalue())
        self.manifest.unlink()  # Recovery uses the saved job snapshot and fingerprints.

        self.commands.clear()
        recovered = R.recover_outputs(self.manifest, run_id)
        self.assertEqual(recovered["synced"], ["second"])
        self.assertEqual(recovered["already_synced"], ["first"])
        self.assertEqual(recovered["pending"], ["third"])
        self.assertEqual(self.commands, ["sessions", "download"])
        _, _, _, remote_file, _ = C.colab.call_args.args
        self.assertEqual(remote_file, f"{C.REMOTE_ROOT}/runs/{run_id}/out/second.png")
        self.assertEqual(C.colab.call_args.kwargs["endpoint"], self.endpoint)
        results = json.loads(self.result_path.read_text())
        self.assertEqual(results["jobs"]["first"], old["jobs"]["first"])
        self.assertEqual(results["jobs"]["second"]["bundle_hash"], self.bundle_hash)
        self.assertTrue(results["jobs"]["second"]["local_synced"])
        self.assertEqual((self.result_path.parent / "second.png").read_bytes(), self.image_bytes)
        self.assertFalse((self.result_path.parent / "unexpected.png").exists())
        self.commands.clear()
        repeat = R.recover_outputs(self.manifest, run_id)
        self.assertEqual(repeat["already_synced"], ["first", "second"])
        self.assertIn("原批次已結束", repeat["errors"]["third"])
        self.assertEqual(self.commands, ["sessions"])

    def test_parallel_runs_merge_different_jobs_and_keep_legacy_results(self):
        out = self.result_path.parent
        out.mkdir()
        legacy = {"status": "done", "file": "out/legacy.png"}
        (out / "legacy.png").write_bytes(_png("green"))
        self.result_path.write_text(json.dumps({"jobs": {"legacy": legacy}, "custom": {"note": "keep"}}))
        other = self._workspace_manifest("other.json", "other", "child_error")
        finish = self._paused_workspace_worker(other, pause_at="download", failed_jobs=("child_error",))

        self.manifest = self._workspace_manifest("manifest.json", "test", "parent_error")
        self.image_bytes = _png("blue")
        status = self.done_status()
        status["jobs"]["parent_error"] = {"status": "error", "error": "parent fixture failure"}
        self.status.return_value = status
        parent = self.run_manifest(force=True)
        outcome, child = finish()
        self.assertEqual(outcome, "ok", child)

        stored = json.loads(self.result_path.read_text())
        self.assertEqual(stored["custom"], {"note": "keep"})
        self.assertEqual(stored["jobs"]["legacy"], legacy)
        self.assertEqual(stored["jobs"]["test"], parent["jobs"]["test"])
        self.assertEqual(stored["jobs"]["other"], child["jobs"]["other"])
        self.assertEqual(set(stored["jobs"]), {"legacy", "test", "other", "child_error", "parent_error"})
        self.assertEqual(set(stored["job_errors"]), {"child_error", "parent_error"})
        self.assertEqual((out / "legacy.png").read_bytes(), _png("green"))
        self.assertEqual((out / "test.png").read_bytes(), _png("blue"))
        self.assertEqual((out / "other.png").read_bytes(), _png("red"))

    def test_parallel_older_run_summary_keeps_newer_image_and_record(self):
        finish = self._paused_workspace_worker(self.manifest, pause_at="finish")
        self.assertEqual(self.result_path.with_name("test.png").read_bytes(), _png("red"))
        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        newer = self.run_manifest(force=True)["jobs"]["test"]
        outcome, child = finish()
        self.assertEqual(outcome, "ok", child)
        self.assertNotEqual(child["run_id"], newer["run_id"])
        self.assertEqual(json.loads(self.result_path.read_text())["jobs"]["test"], newer)
        self.assertEqual(self.result_path.with_name("test.png").read_bytes(), _png("blue"))

    def test_parallel_runs_for_same_job_keep_first_committed_result(self):
        finish = self._paused_workspace_worker(self.manifest, pause_at="download")
        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        committed = self.run_manifest()["jobs"]["test"]
        outcome, delayed = finish()
        self.assertEqual(outcome, "ok", delayed)
        self.assertIn("本機成果紀錄已由其他批次更新", delayed["error"])
        stored = json.loads(self.result_path.read_text())
        self.assertEqual(stored["jobs"]["test"], committed)
        self.assertNotIn("test", stored.get("job_errors", {}))
        self.assertEqual(self.result_path.with_name("test.png").read_bytes(), _png("blue"))

    def test_cross_process_recover_rechecks_same_job_after_download_and_merges_the_rest(self):
        self.manifest = self._workspace_manifest("old.json", "test", "other")
        self.status.return_value = {"done": True, "alive": False, "jobs": {}}
        old_run = self.run_manifest()["run_id"]
        finish = self._paused_workspace_worker(self.manifest, run_id=old_run, pause_at="download")

        self.manifest = self._workspace_manifest("manifest.json", "test")
        self.image_bytes = _png("blue")
        self.status.side_effect = [{"alive": False}, self.done_status()]
        newer = self.run_manifest()["jobs"]["test"]
        outcome, report = finish()
        self.assertEqual(outcome, "ok", report)
        self.assertEqual(report["superseded"], ["test"])
        self.assertEqual(report["synced"], ["other"])
        self.assertEqual(report["errors"], {})
        stored = json.loads(self.result_path.read_text())
        self.assertEqual(stored["jobs"]["test"], newer)
        self.assertEqual(stored["jobs"]["other"]["run_id"], old_run)
        self.assertEqual(self.result_path.with_name("test.png").read_bytes(), _png("blue"))
        self.assertEqual(self.result_path.with_name("other.png").read_bytes(), _png("red"))

    def test_recover_waits_for_cross_process_publication_to_finish(self):
        self.manifest = self._workspace_manifest("old.json", "test")
        self.status.return_value = {"done": True, "alive": False, "jobs": {}}
        old_run = self.run_manifest()["run_id"]
        other = self._workspace_manifest("other.json", "other")
        finish = self._paused_workspace_worker(other, pause_at="publish")

        self.assertEqual(self.result_path.with_name("other.png").read_bytes(), _png("red"))
        self.assertNotIn("other", json.loads(self.result_path.read_text())["jobs"])
        self.commands.clear()
        with patch.object(R, "WORKSPACE_LOCK_TIMEOUT", 0):
            busy = R.recover_outputs(self.manifest, old_run)
        self.assertIn("工作目錄正在發布成果", busy["errors"]["test"])
        self.assertEqual(self.commands, [])
        outcome, child = finish()
        self.assertEqual(outcome, "ok", child)

        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        self.assertEqual(R.recover_outputs(self.manifest, old_run)["synced"], ["test"])
        stored = json.loads(self.result_path.read_text())
        self.assertEqual(stored["jobs"]["other"], child["jobs"]["other"])
        self.assertEqual(stored["jobs"]["test"]["run_id"], old_run)

    def test_parallel_failed_allocation_keeps_other_run_results(self):
        other = self._workspace_manifest("other.json", "other")
        finish = self._paused_workspace_worker(other, pause_at="allocation")
        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        self.run_manifest(stop_after=True)
        committed = self.result_path.read_bytes()
        self.assertEqual(finish(), ("error", "OSError: fixture allocation failed"))
        self.assertEqual(self.result_path.read_bytes(), committed)
        self.assertEqual(self.result_path.with_name("test.png").read_bytes(), _png("blue"))

    def test_recovery_refuses_wrong_account_runtime_run_and_bundle(self):
        self.status.return_value = {"done": True, "alive": False, "jobs": {}}
        run_id = self.run_manifest()["run_id"]

        def recover():
            self.commands.clear()
            C.poll_remote_status.reset_mock()
            report = R.recover_outputs(self.manifest, run_id)
            self.assertNotIn("download", self.commands)
            return report

        C.account_scope.return_value = "foreign"
        self.assertIn("登入身分", recover()["remote_error"])
        self.assertEqual(self.commands, [])
        C.account_scope.return_value = "fixture-account"
        self.runtimes[self.session] = "replacement"
        self.assertEqual(recover()["unavailable"], ["test"])
        C.poll_remote_status.assert_not_called()
        with patch.object(C, "sessions", return_value=([], "")):
            unknown = recover()
        self.assertEqual(unknown["pending"], ["test"])
        self.assertIn("無法確認", unknown["remote_error"])
        self.runtimes[self.session] = self.endpoint

        for wrong in ({"run_id": "f" * 12}, {"bundle_hash": "f" * 64}):
            with self.subTest(wrong=wrong):
                self.status.return_value = dict(self.done_status(), **wrong)
                self.assertIn("批次 ID 或程式版本不符", recover()["remote_error"])
        self.status.return_value = {"stage": "not started", "jobs": {}, "bundle_hash": None}
        self.assertEqual(recover()["unavailable"], ["test"])
        self.status.return_value = None
        self.assertEqual(recover()["pending"], ["test"])
        C.set_keep(self.session, self.now + 30, extra={"release_pending": True})
        self.assertIn("正在關機", recover()["remote_error"])
        C.poll_remote_status.assert_not_called()

    def test_bad_receipts_and_bytes_preserve_the_previous_pair(self):
        old = self.run_manifest()["jobs"]["test"]
        old_image = self.image_bytes
        self.image_bytes = b"new fixture image"
        good = self.done_status()["jobs"]["test"]
        cases = (
            ({key: value for key, value in good.items() if key != "sha256"}, 0, "缺少有效的 SHA-256"),
            (dict(good, sha256="wrong"), 0, "缺少有效的 SHA-256"),
            (dict(good, sha256="f" * 64), 3, "SHA-256 不符"),
            (dict(good, bytes=True), 0, "缺少有效的檔案大小"),
            (dict(good, bytes=1), 3, "檔案大小不符"),
        )
        for info, attempts, message in cases:
            with self.subTest(message=message, info=info):
                self.commands.clear()
                self.status.return_value = dict(self.done_status(), jobs={"test": info})
                failed = self.run_manifest(force=True)
                self.assertEqual(self.commands.count("download"), attempts)
                self.assertIn(message, failed["job_errors"]["test"]["error"])
                self.assertEqual(failed["jobs"]["test"], old)
                self.assertEqual((self.result_path.parent / "test.png").read_bytes(), old_image)
                self.assertIn(message, R.recover_outputs(self.manifest, failed["run_id"])["errors"]["test"])
                self.assertEqual(self.commands.count("download"), attempts + bool(attempts))

    def test_interrupted_download_retries_with_an_independent_file(self):
        destinations = []

        def download(command, *args, **options):
            if command == "download":
                path = Path(args[-1])
                destinations.append(path)
                self.assertFalse(path.exists())
                if len(destinations) <= 3:
                    path.write_bytes(b"partial")
                    raise OSError("download interrupted")
            return self.fake_colab(command, *args, **options)

        C.colab.side_effect = download
        result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "sync_error")
        self.assertFalse((self.result_path.parent / "test.png").exists())
        report = R.recover_outputs(self.manifest, result["run_id"])
        self.assertEqual(report["synced"], ["test"])
        self.assertEqual(len(set(destinations)), 4)
        self.assertFalse(any(path.exists() for path in destinations))
        self.commands.clear()
        self.assertEqual(R.recover_outputs(self.manifest, result["run_id"])["already_synced"], ["test"])
        self.assertEqual(self.commands, [])

    def test_failed_metadata_replace_rolls_back_picture_and_replays_without_runtime(self):
        previous = self.run_manifest()["jobs"]["test"]
        old_image = self.image_bytes
        self.image_bytes = b"new fixture image"
        self.status.return_value = self.done_status()
        replace = Path.replace

        def fail_metadata(path, dest):
            if dest == self.result_path:
                jobs = json.loads(path.read_text())["jobs"]
                if jobs["test"]["sha256"] != previous["sha256"]:
                    raise OSError("results.json replace interrupted")
            return replace(path, dest)

        self.commands.clear()
        with patch.object(Path, "replace", fail_metadata):
            result = self.run_manifest(force=True)
        self.assertEqual(self.commands.count("download"), 1)
        self.assertEqual(json.loads(self.result_path.read_text())["jobs"]["test"], previous)
        self.assertEqual((self.result_path.parent / "test.png").read_bytes(), old_image)
        self.runtimes.clear()
        self.commands.clear()
        self.assertEqual(R.recover_outputs(self.manifest, result["run_id"])["synced"], ["test"])
        self.assertEqual(self.commands, [])
        stored = json.loads(self.result_path.read_text())["jobs"]["test"]
        self.assertEqual(stored["run_id"], result["run_id"])
        self.assertEqual((self.result_path.parent / "test.png").read_bytes(), self.image_bytes)

    def test_first_publication_failure_keeps_the_verified_download_for_retry(self):
        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        save_results = store.save_results

        def fail_metadata(path, data):
            if path == self.result_path and data["jobs"]["test"]["status"] == "done":
                raise OSError("fixture metadata failure")
            save_results(path, data)

        with patch.object(store, "save_results", side_effect=fail_metadata):
            failed = self.run_manifest()
        self.assertFalse(self.result_path.with_name("test.png").exists())
        self.assertFalse(self.result_path.with_name(".publish.json").exists())
        self.assertIn("fixture metadata failure", failed["job_errors"]["test"]["error"])
        self.runtimes.clear()
        self.commands.clear()
        self.assertEqual(R.recover_outputs(self.manifest, failed["run_id"])["synced"], ["test"])
        self.assertEqual(self.result_path.with_name("test.png").read_bytes(), self.image_bytes)
        self.assertEqual(self.commands, [])

    def test_stop_after_image_replace_rolls_back_and_verifies_shutdown(self):
        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        previous = self.run_manifest()["jobs"]["test"]
        self.image_bytes = _png("red")
        self.status.side_effect = [{"alive": False}, self.done_status()]
        image_path = self.result_path.with_name("test.png")
        copy_image = store._copy_image

        def copy_then_stop(source, dest):
            copy_image(source, dest)
            if dest == image_path:
                self.abort = "stop"

        with patch.object(store, "_copy_image", side_effect=copy_then_stop):
            with self.assertRaises(R.RunStopped) as stopped:
                self.run_manifest(force=True)
        self.assertEqual(stopped.exception.outcome, C.ReleaseState.CONFIRMED)
        self.assertIsNone(C.keep_info(self.session))
        self.assertEqual(image_path.read_bytes(), _png("blue"))
        self.assertEqual(json.loads(self.result_path.read_text())["jobs"]["test"], previous)

    def test_worker_exit_after_image_replace_can_finish_from_local_journal(self):
        self.image_bytes = _png("blue")
        self.status.return_value = self.done_status()
        previous = self.run_manifest()["jobs"]["test"]
        old_image = self.image_bytes
        self.image_bytes = _png("red")
        self.status.return_value = self.done_status()
        replace = Path.replace
        image_path = self.result_path.with_name("test.png")

        def fail_image(path, dest):
            if dest == image_path:
                raise OSError("image replace interrupted")
            return replace(path, dest)

        self.manifest.write_text(json.dumps({"jobs": [{"id": "test", "prompt": "a different tree"}]}))
        with patch.object(Path, "replace", fail_image):
            result = self.run_manifest(remake_stale=True)
        self.assertEqual(image_path.read_bytes(), old_image)
        run_id = result["run_id"]
        child = subprocess.run(
            [sys.executable, "-c", """
import os
import sys
from pathlib import Path
from cstudio import results as store

base = Path(sys.argv[1])
run_id = sys.argv[2]
request = store.read_object(base / 'out' / '.runs' / run_id / 'request.json')
outputs = store.RunOutputs(base, run_id, request)
with outputs.lock():
    save_results = store.save_results
    def interrupt(path, data):
        if path == outputs.workspace.path:
            os._exit(73)
        save_results(path, data)
    store.save_results = interrupt
    outputs.publish('test', lambda: None)
""", str(self.directory), run_id],
            check=False,
        )
        self.assertEqual(child.returncode, 73)
        self.assertEqual(image_path.read_bytes(), self.image_bytes)
        self.assertEqual(json.loads(self.result_path.read_text())["jobs"]["test"], previous)
        self.assertEqual((self.result_path.parent / "_stale" / f"test_{run_id}.png").read_bytes(), old_image)

        original_manifest = self.manifest
        self.manifest = self._workspace_manifest("other.json", "other")
        other = dict(self.done_status()["jobs"]["test"], file="out/other.png")
        self.status.side_effect = [{"alive": False}, dict(self.done_status(), jobs={"other": other})]
        other_result = self.run_manifest()["jobs"]["other"]
        self.assertEqual(image_path.read_bytes(), old_image)
        self.assertEqual(json.loads(self.result_path.read_text())["jobs"]["test"], previous)

        self.runtimes.clear()
        self.commands.clear()
        self.assertEqual(R.recover_outputs(original_manifest, run_id)["synced"], ["test"])
        self.assertEqual(self.commands, [])
        stored = json.loads(self.result_path.read_text())["jobs"]
        self.assertEqual(stored["test"]["run_id"], run_id)
        self.assertEqual(stored["other"], other_result)
        self.assertEqual(image_path.read_bytes(), self.image_bytes)

    def test_old_recovery_cannot_replace_a_newer_successful_job(self):
        self.status.return_value = {"done": True, "alive": False, "jobs": {}}
        missed = self.run_manifest()["run_id"]
        self.status.return_value = self.done_status()
        new = self.run_manifest()["jobs"]["test"]
        self.commands.clear()
        self.assertEqual(R.recover_outputs(self.manifest, missed)["superseded"], ["test"])
        self.assertEqual(self.commands, [])
        self.assertEqual(json.loads(self.result_path.read_text())["jobs"]["test"], new)

    def test_recovery_rechecks_runtime_identity_after_download(self):
        self.status.return_value = {"done": True, "alive": False, "jobs": {}}
        run_id = self.run_manifest()["run_id"]
        self.status.return_value = self.done_status()

        def download(command, *args, **options):
            result = self.fake_colab(command, *args, **options)
            if command == "download":
                self.runtimes[self.session] = "replacement"
            return result

        C.colab.side_effect = download
        report = R.recover_outputs(self.manifest, run_id)
        self.assertIn("識別不符", report["errors"]["test"])
        self.assertFalse((self.result_path.parent / "test.png").exists())
        self.assertFalse((self.result_path.parent / ".runs" / run_id / ".result-test.json").exists())

    def test_pinned_resume_verifies_execution_and_picture_without_colab(self):
        original = self.run_manifest()
        saved = original["jobs"]["test"]
        self.assertEqual(saved["fingerprint_version"], 2)
        self.assertEqual(saved["execution"]["environment"]["packages"]["torch"], "2.7")
        self.assertNotEqual(saved["fingerprint"], saved["execution_fingerprint"])
        self.commands.clear()
        with patch.object(C, "token_ok", return_value=False) as token:
            self.assertEqual(self.run_manifest(), original)
        self.assertEqual(self.commands, [])
        token.assert_not_called()

        manifest = json.loads(self.manifest.read_text())
        for changes in ({"gpu": "A100"}, {"pin_versions": False}, {"pin_models": False}, {"bundle_hash": "0" * 64}):
            with self.subTest(changes=changes):
                fingerprints, _, done, stale = R.classify_jobs(manifest, self.directory, **changes)
                self.assertNotEqual(fingerprints["test"], saved["fingerprint"])
                self.assertFalse(done)
                self.assertIn("執行設定已變更", stale[0][1])

        saved["execution"]["environment"]["packages"]["torch"] = "changed"
        self.result_path.write_text(json.dumps(original))
        _, _, done, stale = R.classify_jobs(manifest, self.directory)
        self.assertFalse(done)
        self.assertEqual(stale, [("test", "本機成果的執行環境指紋不符")])

    def test_execution_fingerprint_changes_with_actual_environment_and_fitted_paths(self):
        original = self.run_manifest()["jobs"]["test"]
        snapshot = self.done_status()
        actual = snapshot["jobs"]["test"]["execution"]
        actual["environment"]["packages"]["torch"] = "2.8"
        actual["models"][0]["path"] = "models/diffusion_models/model.gguf"
        self.status.return_value = snapshot
        changed = self.run_manifest(force=True)["jobs"]["test"]
        self.assertEqual(changed["fingerprint"], original["fingerprint"])
        self.assertNotEqual(changed["execution_fingerprint"], original["execution_fingerprint"])
        self.assertEqual(changed["execution"], actual)
        actual["pin_models"] = False
        unfitted = self.run_manifest(force=True, pin_models=False)["jobs"]["test"]
        actual["fitted_models"] = [["10", "unet_name", "model.gguf", "other/model.gguf"]]
        fitted = self.run_manifest(force=True, pin_models=False)["jobs"]["test"]
        self.assertEqual(fitted["fingerprint"], unfitted["fingerprint"])
        self.assertNotEqual(fitted["execution_fingerprint"], unfitted["execution_fingerprint"])

    def test_resume_detects_same_size_output_changes_even_with_no_fingerprint(self):
        original = self.run_manifest()["jobs"]["test"]
        output = self.result_path.with_name("test.png")
        changed = b"z" * original["bytes"]
        output.write_bytes(changed)
        self.commands.clear()
        for use_fingerprint in (True, False):
            with self.subTest(use_fingerprint=use_fingerprint):
                with self.assertRaisesRegex(RuntimeError, "大小或 SHA-256 已變更"):
                    self.run_manifest(use_fingerprint=use_fingerprint)
                self.assertEqual(self.commands, [])
                self.assertEqual(output.read_bytes(), changed)

        reworked = self.run_manifest(remake_stale=True, stop_after=True)
        self.assertEqual(output.read_bytes(), self.image_bytes)
        self.assertEqual(reworked["shutdown_state"], C.ReleaseState.CONFIRMED)
        self.assertEqual((output.parent / "_stale" / f"test_{reworked['run_id']}.png").read_bytes(), changed)

    def test_unpinned_output_requires_explicit_reuse_even_when_request_matches(self):
        snapshot = self.done_status()
        actual = snapshot["jobs"]["test"]["execution"]
        actual.update(pin_versions=False, pin_models=False)
        actual["models"][0].pop("sha256")
        self.status.return_value = snapshot
        with patch.object(R, "bundle_and_launch_code", wraps=R.bundle_and_launch_code) as prepare:
            original = self.run_manifest(pin_versions=False)
        sent = json.loads(prepare.call_args.args[2])
        self.assertFalse(sent["_pin_versions"])
        self.assertFalse(sent["_pin_models"])
        self.assertEqual(original["jobs"]["test"]["execution"], actual)
        self.commands.clear()
        with self.assertRaisesRegex(RuntimeError, "執行環境未完整釘選或驗證"):
            self.run_manifest(pin_versions=False)
        self.assertEqual(self.run_manifest(pin_versions=False, use_fingerprint=False), original)
        self.assertEqual(self.commands, [])

    def test_recovery_accepts_pre_upgrade_request_without_execution_receipt(self):
        self.status.return_value = {"done": True, "alive": False, "jobs": {}}
        run_id = self.run_manifest()["run_id"]
        request_path = self.result_path.parent / ".runs" / run_id / "request.json"
        request = json.loads(request_path.read_text())
        request["schema"] = 1
        request.pop("pin_versions")
        request.pop("pin_models")
        manifest = json.loads(self.manifest.read_text())
        job = manifest["jobs"][0]
        request["jobs"]["test"]["fingerprint"] = R.compute_job_fingerprint(
            job, manifest, self.directory, versions=R.load_versions(),
        )
        request_path.write_text(json.dumps(request))
        snapshot = self.done_status()
        snapshot["jobs"]["test"].pop("execution")
        self.status.return_value = snapshot
        self.assertEqual(R.recover_outputs(self.manifest, run_id)["synced"], ["test"])
        self.assertEqual(R.plan_batch(self.manifest)["legacy"], ["test"])
        self.assertNotIn("execution_fingerprint", json.loads(self.result_path.read_text())["jobs"]["test"])

    def test_legacy_output_without_digest_requires_explicit_reuse(self):
        self.result_path.parent.mkdir()
        self.result_path.with_name("test.png").write_bytes(self.image_bytes)
        manifest = json.loads(self.manifest.read_text())
        fingerprint = R.compute_job_fingerprint(
            manifest["jobs"][0], manifest, self.directory, versions=R.load_versions(),
        )
        record = {"jobs": {"test": {"fingerprint": fingerprint, "status": "done"}}}
        self.result_path.write_text(json.dumps(record))
        plan = R.plan_batch(self.manifest)
        self.assertEqual(plan["stale"], [("test", "舊成果缺少大小與 SHA-256 紀錄")])
        self.assertEqual(self.run_manifest(use_fingerprint=False), record)
        self.assertEqual(self.commands, [])

    def test_verified_image_without_full_serving_versions_cannot_auto_resume(self):
        snapshot = self.done_status()
        del snapshot["jobs"]["test"]["execution"]["comfy"]["versions"]["pytorch_version"]
        self.status.return_value = snapshot
        saved = self.run_manifest()
        self.assertTrue(saved["jobs"]["test"]["local_synced"])
        self.commands.clear()
        with self.assertRaisesRegex(RuntimeError, "ComfyUI 未回報完整版本"):
            self.run_manifest()
        self.assertEqual(self.run_manifest(use_fingerprint=False), saved)
        self.assertEqual(self.commands, [])

    def test_new_run_cannot_publish_a_receipt_without_matching_execution(self):
        snapshot = self.done_status()
        info = snapshot["jobs"]["test"]
        info["execution"]["pin_models"] = False
        self.status.return_value = snapshot
        result = self.run_manifest()
        self.assertIn("缺少與批次相符的執行環境紀錄", result["job_errors"]["test"]["error"])
        self.assertNotIn("download", self.commands)
        self.assertFalse(self.result_path.with_name("test.png").exists())

    def test_confirmed_cancel_limits_the_total_time_spent_on_downloads(self):
        self.manifest.write_text(json.dumps({"jobs": [
            {"id": jid, "prompt": jid} for jid in ("first", "second")
        ]}), encoding="utf-8")
        info = self.done_status()["jobs"]["test"]
        self.status.return_value = dict(self.done_status(), stage="cancelled", cancel_protocol=1, jobs={
            jid: dict(info, file=f"out/{jid}.png") for jid in ("first", "second")
        })

        def download(command, *args, **options):
            if command == "download":
                self.assertEqual(options["deadline"], self.now + C.RECOVERY_TIMEOUT)
                self.advance(C.RECOVERY_TIMEOUT)
                self.commands.append(command)
                raise C.ColabDeadline("download deadline")
            return self.fake_colab(command, *args, **options)

        C.colab.side_effect = download
        with self.assertRaises(R.RunCancelled):
            self.run_manifest()
        self.assertEqual(self.commands.count("download"), 1)
        self.assertNotIn("stop", self.commands)
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 30 * 60)

    def test_stop_during_download_does_not_publish_the_picture(self):
        def download(command, *args, **options):
            result = self.fake_colab(command, *args, **options)
            if command == "download":
                self.abort = "stop"
            return result

        C.colab.side_effect = download
        with self.assertRaises(R.RunStopped) as stopped:
            self.run_manifest()
        self.assertEqual(stopped.exception.outcome, C.ReleaseState.CONFIRMED)
        self.assertFalse((self.result_path.parent / "test.png").exists())
        self.assertEqual(self.commands.count("download"), 1)
        run_id = json.loads(self.result_path.read_text())["run_id"]
        self.assertEqual(R.recover_outputs(self.manifest, run_id)["unavailable"], ["test"])

    def test_reuse_owned_runtime_keeps_its_bundle_cache(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now + 60, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "bundle_hash": R.compute_remote_bundle_hash(),
        })
        self.status.side_effect = [{"alive": False}, self.done_status()]
        with patch.object(R, "bundle_and_launch_code", wraps=R.bundle_and_launch_code) as bundle:
            result = self.run_manifest()
        self.assertEqual(result["jobs"]["test"]["status"], "done")
        self.assertIsNone(bundle.call_args.args[0])
        self.assertEqual(C.keep_info(self.session)["runtime_endpoint"], self.endpoint)
        self.assertNotIn("new", self.commands)
        self.assertNotIn("stop", self.commands)

    def test_reuse_external_runtime_never_arms_a_hard_deadline(self):
        self.runtimes[self.session] = self.endpoint
        self.status.side_effect = [{"alive": False}, self.done_status()]
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
            "busy_until": self.now + 30, "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        C.watchdog_loop(self.session)
        self.assertNotIn(self.session, self.runtimes)
        self.assertEqual(self.commands, ["sessions", "stop", "sessions"])
        C.poll_remote_status.assert_not_called()
        self.assertIsNone(C.keep_info(self.session))

    def test_watchdog_does_not_stop_after_keep_extended_during_listing(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now - 1, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        snapshot = C.keep_info(self.session)

        def extend(**_options):
            C.set_keep(self.session, self.now + 600)
            return [self.session], self.fake_colab("sessions")

        with patch.object(C, "sessions_checked", side_effect=extend):
            C.release_runtime(self.session, watchdog_keep=snapshot, account="fixture-account")
        self.assertNotIn("stop", self.commands)
        self.assertIn(self.session, self.runtimes)
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 600)

    def test_watchdog_claim_prevents_extension_once_stop_dispatch_begins(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now - 1, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        snapshot = C.keep_info(self.session)

        def check_dispatch():
            C.set_keep(self.session, self.now + 600)
            record = C.keep_info(self.session)
            self.assertTrue(record.get("release_pending"))
            self.assertLess(record["expires"], self.now + 600)

        self.on_stop.side_effect = check_dispatch
        self.assertEqual(
            C.release_runtime(self.session, watchdog_keep=snapshot, account="fixture-account"),
            C.ReleaseState.CONFIRMED,
        )
        self.on_stop.assert_called_once()

    def test_idle_watchdog_allows_active_batch_only_through_bounded_grace(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, self.now, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        self.status.return_value = {"alive": True}
        limit = self.now + C.IDLE_ACTIVE_GRACE
        self.on_stop.side_effect = lambda: self.assertEqual(self.now, limit)
        C.watchdog_loop(self.session)
        self.assertGreater(C.poll_remote_status.call_count, 0)
        self.on_stop.assert_called_once()

    def test_extended_keep_does_not_override_hard_deadline_of_active_run(self):
        self.runtimes[self.session] = self.endpoint
        hard_limit = self.now + 90
        C.set_keep(self.session, self.now + 600, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
            "active_run_id": "active-run", "busy_until": hard_limit,
        })
        self.on_stop.side_effect = lambda: self.assertEqual(self.now, hard_limit)
        C.watchdog_loop(self.session)
        self.on_stop.assert_called_once()
        C.poll_remote_status.assert_not_called()

    def test_legacy_busy_deadline_protects_run_if_idle_expiry_is_earlier(self):
        self.runtimes[self.session] = self.endpoint
        hard_limit = self.now + 90
        C.set_keep(self.session, self.now + 30, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account", "busy_until": hard_limit,
        })
        self.on_stop.side_effect = lambda: self.assertEqual(self.now, hard_limit)
        C.watchdog_loop(self.session)
        self.on_stop.assert_called_once()
        C.poll_remote_status.assert_not_called()

    def test_active_run_record_blocks_reuse_before_it_is_observable_remotely(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, None, owner="owned", extra={
            "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
            "active_run_id": "other-run", "busy_until": self.now + 600,
        })
        original = C.keep_info(self.session)
        with self.assertRaisesRegex(RuntimeError, "已有批次的執行紀錄"):
            self.run_manifest()
        self.assertEqual(C.keep_info(self.session), original)
        C.poll_remote_status.assert_not_called()
        C.remote_exec.assert_not_called()

    def test_runner_preserves_keep_extended_while_it_finishes(self):
        extended = self.now + 4_000

        def status(_session, **_target):
            C.set_keep(self.session, extended)
            return self.done_status()

        self.status.side_effect = status
        self.run_manifest()
        keep = C.keep_info(self.session)
        self.assertEqual(keep["expires"], extended)
        self.assertIsNone(keep["active_run_id"])
        self.assertIsNone(keep["busy_until"])

    def test_watchdog_waits_for_original_account_before_reconciling(self):
        self.runtimes[self.session] = self.endpoint
        C.set_keep(self.session, None, owner="owned", extra={
            "busy_until": self.now, "runtime_endpoint": self.endpoint, "account_scope": "fixture-account",
        })
        C.account_scope.return_value = "foreign-account"

        def tick(seconds):
            self.advance(seconds)
            if self.now == 1_330:
                C.colab.assert_not_called()
                self.assertEqual(C.keep_info(self.session)["runtime_endpoint"], self.endpoint)
                C.account_scope.return_value = "fixture-account"

        with patch.object(C.time, "sleep", side_effect=tick):
            C.watchdog_loop(self.session)
        self.assertEqual(self.commands, ["sessions", "stop", "sessions"])
        self.assertIsNone(C.keep_info(self.session))

    def test_watchdog_preserves_unconfirmed_allocation_until_verified_absent(self):
        C.set_keep(self.session, None, owner="pending", extra={
            "busy_until": self.now, "account_scope": "fixture-account", "allocation_baseline": [],
        })
        listings = iter([
            ([], "[?] fixture-runtime | Hardware: L4"),
            (["foreign"], "[foreign] fixture-runtime | Hardware: L4"),
            ([], self.no_sessions),
        ])

        def list_while_pending(**options):
            self.assertEqual(C.keep_info(self.session)["created_by"], "pending")
            return next(listings)

        with patch.object(C, "sessions_checked", side_effect=list_while_pending) as checks:
            C.watchdog_loop(self.session)
        self.assertEqual(checks.call_count, 3)
        self.assertNotIn("stop", self.commands)
        self.assertIsNone(C.keep_info(self.session))

    def test_release_does_not_stop_a_different_runtime_with_same_name(self):
        self.runtimes[self.session] = "replacement-runtime"
        C.set_keep(self.session, self.now + 60, owner="external", extra={
            "runtime_endpoint": "replacement-runtime", "account_scope": "fixture-account",
        })
        self.assertEqual(C.release_runtime(self.session, "owned", self.endpoint), C.ReleaseState.CONFIRMED)
        self.assertEqual(self.commands, ["sessions"])
        self.assertEqual(C.keep_info(self.session)["runtime_endpoint"], "replacement-runtime")

    def test_cancel_during_retry_backoff_prevents_second_allocation(self):
        abort = Mock(side_effect=[None, None, RuntimeError("fixture cancelled")])
        with patch.object(C, "colab", side_effect=RuntimeError(self.allocation_refused)) as command:
            with self.assertRaisesRegex(RuntimeError, "fixture cancelled"):
                C.new_runtime(self.session, "L4", retries=2, wait=3, check_abort=abort)
        command.assert_called_once()
        self.assertEqual(self.now, 1_003.0)

    def test_allocation_deadline_bounds_request_and_backoff(self):
        with patch.object(C, "colab", side_effect=RuntimeError(self.allocation_refused)) as command:
            with self.assertRaisesRegex(C.ColabDeadline, "批次時間上限"):
                C.new_runtime(self.session, "L4", wait=120, deadline=self.now + 12)
        self.assertEqual(command.call_args_list, [
            call("new", "-s", self.session, "--gpu", "L4", timeout=12, account="fixture-account", deadline=1_012),
        ])
        self.assertEqual(self.now, 1_012.0)

    def test_subprocess_timeout_is_clipped_by_monotonic_deadline(self):
        error = subprocess.TimeoutExpired(["fake-colab"], timeout=3)
        with patch.object(C, "colab", self.real_colab), patch.object(C.subprocess, "run", side_effect=error) as run:
            with self.assertRaises(C.ColabDeadline):
                C.colab("download", timeout=600, check=False, account="fixture-account", deadline=self.now + 3)
        self.assertEqual(run.call_args.kwargs["timeout"], 3)

    def test_work_deadline_ignores_wall_clock_rewind_and_cleanup_has_own_budget(self):
        def rewind_and_finish_allocation():
            self.now += 61
            self.clock.time = lambda: self.now - 10_000

        self.on_allocate.side_effect = rewind_and_finish_allocation
        with self.assertRaisesRegex(RuntimeError, "分鐘上限"):
            self.run_manifest(max_minutes=1)
        new = next(item for item in C.colab.call_args_list if item.args[0] == "new")
        stop = next(item for item in C.colab.call_args_list if item.args[0] == "stop")
        self.assertEqual(new.kwargs["deadline"], 1_060)
        self.assertEqual(stop.kwargs["deadline"], 1_061 + C.RECOVERY_TIMEOUT)
        self.assertIsNone(C.keep_info(self.session))

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
            return self.prep_reply()

        def snapshot(_session, **_target):
            run_id = progress.call_args.args[0]["run_id"]
            if C.poll_remote_status.call_count == 1:
                return {"run_id": run_id, "stage": "running", "done": False}
            return dict(self.done_status(), run_id=run_id, stage="cancelled", cancel_protocol=1)

        C.remote_exec.side_effect = prepare
        self.status.side_effect = snapshot
        with patch.object(C, "request_cancel", return_value={"state": "pending"}) as send:
            with self.assertRaisesRegex(R.RunCancelled, "已確認遠端批次取消"):
                self.run_manifest(progress_cb=progress)
        cancelled = progress.call_args.args[0]
        send.assert_called_once_with(self.session, cancelled["run_id"], allow_start=True,
                                     account="fixture-account", endpoint=self.endpoint, deadline=8_200)
        self.assertEqual(cancelled["stage"], "cancelled")
        self.assertEqual(cancelled["_local_synced_jobs"], ["test"])
        self.assertEqual((self.result_path.parent / "test.png").read_bytes(), self.image_bytes)
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 30 * 60)
        self.assertNotIn("stop", self.commands)

    def test_failed_remote_cancel_preserves_existing_recovery_deadline(self):
        progress = Mock()

        def prepare(_session, _code, **_opts):
            self.abort = "cancel"
            return self.prep_reply()

        C.remote_exec.side_effect = prepare
        self.status.side_effect = lambda _session, **_target: {
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
        self.status.return_value = {"stage": "cancel_failed", "alive": False}
        with self.assertRaisesRegex(RuntimeError, "待確認的取消"):
            self.run_manifest()
        C.remote_exec.assert_not_called()
        self.assertNotIn("new", self.commands)
        self.assertNotIn("stop", self.commands)

    def test_lost_cancel_reply_retries_without_treating_it_as_confirmation(self):
        progress = Mock()
        C.remote_exec.side_effect = lambda *_args, **_opts: (
            setattr(self, "abort", "cancel") or self.prep_reply()
        )

        def snapshot(_session, **_target):
            run_id = progress.call_args.args[0]["run_id"]
            if C.poll_remote_status.call_count == 1:
                return {"run_id": run_id, "stage": "running", "done": False}
            return dict(self.done_status(), run_id=run_id)

        self.status.side_effect = snapshot
        with patch.object(C, "request_cancel", side_effect=[RuntimeError("lost reply"), {"state": "finished"}]) as send:
            result = self.run_manifest(progress_cb=progress)
        self.assertEqual(send.call_count, 2)
        self.assertNotIn("error", result)
        self.assertEqual(progress.call_args.args[0]["stage"], "done")
        self.assertEqual(C.keep_info(self.session)["expires"], self.now + 15 * 60)


if __name__ == "__main__":
    unittest.main()
