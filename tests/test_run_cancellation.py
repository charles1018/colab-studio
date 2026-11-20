"""Offline tests for the local and remote ends of run-scoped cancellation."""
import io
import json
import signal
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from fastapi import HTTPException

from cstudio import cli, colab_client, webui
from remote import cancel, remote_runner as remote


RUN = "0123456789ab"
OTHER_RUN = "0123456789ac"
PROMPT = "d258704a-878a-4f8d-8ff0-f2c5e0de7bf2"
OTHER_PROMPT = "92b82ed7-5286-4cb4-becd-3777a17a78b7"


class RemoteCancellationTests(unittest.TestCase):
    def setUp(self):
        self.job = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.status = self.job / "status.json"
        self.state = {"run_id": RUN, "log": [], "jobs": {}, "prompt_ids": [], "cancel_protocol": 1}
        self.now = 0
        self.install = Mock()
        self.download = Mock()
        self.download_models = remote.download_models
        self.start_comfy = Mock()
        self.comfy = Mock(side_effect=AssertionError("ComfyUI network access blocked"))
        clock = SimpleNamespace(
            time=lambda: self.now, monotonic=lambda: self.now,
            sleep=self.advance, strftime=lambda _: "00:00:00",
        )
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.multiple(
            remote, JOB=self.job, STATUS=self.status, OUT=self.job / "out",
            WORK=self.job / "work", STATE=self.state, IN_FLIGHT={}, FITTED={},
            install=self.install, download_models=self.download, start_comfy=self.start_comfy,
            call=self.comfy, time=clock,
        ))
        self.status.write_text(json.dumps(self.state), encoding="utf-8")
        (self.job / "manifest.json").write_text(
            json.dumps({"_run_id": RUN, "jobs": [{"id": "a", "prompt": "test"}]}), encoding="utf-8"
        )

    def advance(self, seconds):
        self.now += seconds

    def marker(self, run_id=RUN):
        return cancel.cancel_marker(self.job, run_id)

    def test_marker_targets_expected_run_and_does_not_write_worker_status(self):
        self.assertEqual(cancel.request_cancel(self.job, OTHER_RUN)["state"], "idle")
        self.assertFalse(self.marker(OTHER_RUN).exists())
        before = self.status.read_bytes()
        for _ in range(2):
            self.assertEqual(cancel.request_cancel(self.job, RUN), {"run_id": RUN, "state": "pending"})
        self.assertTrue(self.marker().exists())
        self.assertEqual(self.status.read_bytes(), before)

    def test_legacy_active_worker_reports_unsupported(self):
        self.status.write_text(json.dumps({"run_id": RUN}), encoding="utf-8")
        self.assertEqual(cancel.request_cancel(self.job, None)["state"], "unsupported")
        self.assertFalse(self.marker().exists())

    def test_request_before_start_skips_setup(self):
        self.status.unlink()
        self.assertEqual(cancel.request_cancel(self.job, RUN, allow_start=True)["state"], "pending")
        remote.main()
        self.install.assert_not_called()
        self.comfy.assert_not_called()
        result = json.loads(self.status.read_text(encoding="utf-8"))
        self.assertTrue(result["done"])
        self.assertEqual(result["stage"], "cancelled")
        self.assertEqual(cancel.request_cancel(self.job, RUN)["state"], "confirmed")

    def test_cancellation_during_install_stops_before_models_or_comfy(self):
        self.install.side_effect = lambda _: cancel.request_cancel(self.job, RUN)
        remote.main()
        self.download.assert_not_called()
        self.start_comfy.assert_not_called()
        self.assertEqual(self.state["stage"], "cancelled")

    @unittest.skipUnless(hasattr(remote.os, "killpg"), "Colab setup subprocesses run on POSIX")
    def test_cancel_sends_signals_only_to_owned_setup_process_group(self):
        process = MagicMock(pid=4321)
        process.__enter__.return_value = process
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["fixture"], 1),
            subprocess.TimeoutExpired(["fixture"], 5),
            ("", None),
        ]
        start = Mock(return_value=process)

        def start_then_cancel(*args, **opts):
            self.assertTrue(opts["start_new_session"])
            cancel.request_cancel(self.job, RUN)
            return start(*args, **opts)

        with patch.object(remote.subprocess, "Popen", side_effect=start_then_cancel), \
             patch.object(remote.os, "killpg") as terminate:
            with self.assertRaises(remote.BatchCancelled):
                remote.setup_command(["fixture"])
        self.assertEqual(terminate.call_args_list, [
            call(process.pid, signal.SIGTERM),
            call(process.pid, signal.SIGKILL),
        ])
        self.start_comfy.assert_not_called()

    def test_model_transfer_keeps_existing_revision_and_cache_destination(self):
        source = self.job / "cached.gguf"
        source.write_bytes(b"blob")
        pins = {"models": [{
            "repo": "fixture/model", "file": "folder/model.gguf", "bytes": 4, "revision": "fixture-revision",
        }]}
        transfer = Mock(return_value=SimpleNamespace(stdout="CSTUDIO_MODEL " + json.dumps(str(source))))
        with patch.object(remote, "COMFY", self.job / "comfy"), \
             patch.object(remote, "setup_command", transfer), \
             patch.object(remote.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10)), \
             patch.dict(remote.os.environ):
            self.download_models([("fixture/model", "folder/model.gguf", "unet")], pins)
        self.assertEqual(transfer.call_args.args[0][-3:], ["fixture/model", "folder/model.gguf", "fixture-revision"])
        self.assertEqual((self.job / "comfy/models/unet/model.gguf").read_bytes(), b"blob")

    def test_targeted_cancel_waits_until_own_prompt_leaves_running_queue(self):
        remote.IN_FLIGHT[PROMPT] = True
        self.comfy.side_effect = [
            {"queue_running": [[0, PROMPT]], "queue_pending": [[1, OTHER_PROMPT]]},
            {"cancelled": True},
            {"queue_running": [[1, OTHER_PROMPT]], "queue_pending": []},
        ]
        remote.cancel_prompts()
        base = remote.COMFY_URL
        self.assertEqual(self.comfy.call_args_list, [
            call(base + "/queue", timeout=5),
            call(base + f"/api/jobs/{PROMPT}/cancel", {}, timeout=5),
            call(base + "/queue", timeout=5),
        ])
        self.assertEqual(remote.IN_FLIGHT, {})

    def test_pending_prompt_uses_the_same_targeted_endpoint(self):
        remote.IN_FLIGHT[PROMPT] = True
        self.comfy.side_effect = [
            {"queue_running": [[0, OTHER_PROMPT]], "queue_pending": [[1, PROMPT]]},
            {"cancelled": True},
            {"queue_running": [[0, OTHER_PROMPT]], "queue_pending": []},
        ]
        remote.cancel_prompts()
        self.assertEqual(self.comfy.call_args_list[1], call(remote.COMFY_URL + f"/api/jobs/{PROMPT}/cancel", {}, timeout=5))

    def test_successful_cancel_rpc_does_not_confirm_prompt_that_stays_active(self):
        remote.IN_FLIGHT[PROMPT] = True
        self.comfy.side_effect = lambda url, *_args, **_opts: (
            {"queue_running": [[0, PROMPT]], "queue_pending": []}
            if url.endswith("/queue") else {"cancelled": True}
        )
        with self.assertRaises(remote.CancellationUnconfirmed):
            remote.cancel_prompts()

    def test_missing_prompt_during_uncertain_submission_is_not_confirmation(self):
        remote.IN_FLIGHT[PROMPT] = False
        self.comfy.side_effect = lambda url, *_args, **_opts: (
            {"queue_running": [], "queue_pending": []} if url.endswith("/queue") else {}
        )
        with self.assertRaises(remote.CancellationUnconfirmed):
            remote.cancel_prompts()

    def test_malformed_queue_does_not_confirm(self):
        remote.IN_FLIGHT[PROMPT] = True
        self.comfy.side_effect = None
        self.comfy.return_value = {"queue_running": "placeholder", "queue_pending": []}
        with self.assertRaises(remote.CancellationUnconfirmed):
            remote.cancel_prompts()

    def test_submission_id_exists_before_request_reaches_comfy(self):
        def submit(_url, payload):
            self.assertEqual(remote.IN_FLIGHT[payload["prompt_id"]], False)
            self.assertEqual(self.state["prompt_ids"], [payload["prompt_id"]])
            raise TimeoutError("fixture lost reply")

        self.comfy.side_effect = submit
        with patch.object(remote, "fit_names", return_value=[]):
            with self.assertRaises(TimeoutError):
                remote.queue_prompt({})
        self.assertEqual(len(remote.IN_FLIGHT), 1)

    def test_cancel_preserves_successful_history_when_prompt_already_finished(self):
        cancel.request_cancel(self.job, RUN)
        remote.IN_FLIGHT[PROMPT] = True
        history = {"status": {"completed": True, "status_str": "success"}}
        self.comfy.side_effect = [{"queue_running": [], "queue_pending": []}, {PROMPT: history}]
        self.assertEqual(remote.wait_history(PROMPT, 100, "fixture"), history)

    def test_cancel_during_job_collection_preserves_completed_job(self):
        original = json.loads((self.job / "manifest.json").read_text(encoding="utf-8"))
        original["jobs"].append({"id": "b", "prompt": "test"})
        (self.job / "manifest.json").write_text(json.dumps(original), encoding="utf-8")

        def collect(_history, dest, **_save_image):
            dest.write_bytes(b"fixture")
            cancel.request_cancel(self.job, RUN)

        with patch.multiple(
            remote, queue_prompt=Mock(return_value=PROMPT), wait_history=Mock(return_value={}),
            collect_output_image=Mock(side_effect=collect),
        ):
            remote.main()
            remote.queue_prompt.assert_called_once()
        self.assertEqual(self.state["jobs"]["a"]["status"], "done")
        self.assertNotIn("b", self.state["jobs"])
        self.assertEqual(self.state["stage"], "cancelled")

    def test_cancel_after_last_output_was_saved_reports_completed_batch(self):
        def collect(_history, dest, **_save_image):
            dest.write_bytes(b"fixture")
            cancel.request_cancel(self.job, RUN)

        with patch.multiple(
            remote, queue_prompt=Mock(return_value=PROMPT), wait_history=Mock(return_value={}),
            collect_output_image=Mock(side_effect=collect),
        ):
            remote.main()
        self.assertEqual(self.state["jobs"]["a"]["status"], "done")
        self.assertEqual(self.state["stage"], "done")
        self.assertEqual(cancel.request_cancel(self.job, RUN)["state"], "finished")


class CancellationClientTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(colab_client, "KEEP", Path(directory) / "keep"))
        self.enterContext(patch.object(colab_client, "token_ok", return_value=False))
        self.state = webui.StudioUIState()
        self.enterContext(patch.object(webui, "STATE", self.state))
        self.state.running = True
        self.state.run_id = RUN
        self.state.session = "local-run"
        self.state.stage = "running"

    def test_matching_local_run_acknowledges_pending_without_remote_call(self):
        with patch.object(colab_client, "request_cancel") as send:
            response = webui.api_colab_cancel({"session": "local-run", "run_id": RUN})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(response.body)["state"], "pending")
        self.assertEqual(self.state.stage, "cancelling")
        send.assert_not_called()

    def test_another_session_does_not_cancel_the_local_run(self):
        with patch.object(colab_client, "request_cancel", return_value={"state": "pending", "run_id": OTHER_RUN}) as send:
            response = webui.api_colab_cancel({"session": "other", "run_id": OTHER_RUN})
        self.assertEqual(response.status_code, 202)
        send.assert_called_once_with("other", OTHER_RUN)
        self.assertFalse(self.state.cancel_requested)
        self.assertEqual(self.state.stage, "running")

    def test_stale_ui_run_does_not_cancel_replacement(self):
        with self.assertRaises(HTTPException) as error:
            webui.api_colab_cancel({"session": "local-run", "run_id": OTHER_RUN})
        self.assertEqual(error.exception.status_code, 409)
        self.assertFalse(self.state.cancel_requested)

    def test_transport_failure_cannot_confirm_or_extend_keep(self):
        with patch.object(colab_client, "request_cancel", side_effect=RuntimeError("transport failure")):
            with self.assertRaises(HTTPException) as error:
                webui.api_colab_cancel({"session": "other"})
        self.assertEqual(error.exception.status_code, 502)
        self.assertIsNone(colab_client.keep_info("other"))

    def test_local_client_rejects_confirmation_for_another_run(self):
        output = f'CSTUDIO_CANCEL {{"state": "confirmed", "run_id": "{OTHER_RUN}"}}'
        with patch.multiple(colab_client, remote_exec=Mock(return_value=output),
                            account_scope=Mock(return_value="fixture-account"), active_endpoint=Mock(return_value="endpoint")):
            with self.assertRaisesRegex(RuntimeError, "遠端未回傳"):
                colab_client.request_cancel("other", RUN)

    def test_cli_reports_pending_without_extending_keep(self):
        with patch.multiple(
            colab_client, sessions=Mock(return_value=(["other"], "")), account_scope=Mock(return_value="fixture-account"),
            request_cancel=Mock(return_value={"state": "pending", "run_id": RUN}),
        ), redirect_stdout(io.StringIO()) as printed:
            cli.cmd_cancel(Namespace(session="other", run_id=RUN))
        self.assertIn("等待遠端確認", printed.getvalue())
        self.assertIsNone(colab_client.keep_info("other"))


if __name__ == "__main__":
    unittest.main()
