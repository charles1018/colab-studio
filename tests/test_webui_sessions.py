"""Offline workbench status checks with controlled Colab replies and clocks."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from fastapi import HTTPException

from cstudio import colab_client as C
from cstudio import webui as W


class WebSessionTests(unittest.TestCase):
    def setUp(self):
        self.account = "fixture-account"
        self.wall = 10_000.0
        self.mono = 100.0
        self.runtimes = {}
        self.enterContext(patch.object(
            W, "time", SimpleNamespace(time=lambda: self.wall, monotonic=lambda: self.mono),
        ))
        self.enterContext(patch.multiple(
            C, KEEP=Path(self.enterContext(tempfile.TemporaryDirectory())) / "keep",
            account_scope=Mock(side_effect=lambda: self.account),
            colab=Mock(side_effect=AssertionError("No live Colab commands")),
            usage=Mock(return_value=(27.0, "27 compute units")),
            sessions=Mock(side_effect=self.listing),
        ))
        self.ui = W.StudioUIState()

    def listing(self, *, account, check):
        self.assertTrue(check)
        return [name for name in self.runtimes if name != "?"], "\n".join(
            f"[{name}] {endpoint} | Hardware: {gpu} | Shape: Standard | Variant: GPU"
            for name, (endpoint, gpu) in self.runtimes.items()
        ) or "[colab] No active sessions found on server."

    def lease(self, name, endpoint, *, expires=None, owner="owned", account="fixture-account", **extra):
        C.set_keep(name, expires, owner=owner, extra=dict(
            account_scope=account, runtime_endpoint=endpoint, **extra,
        ))

    def test_external_assignment_survives_fast_polls_and_unknown_then_confirms_absence(self):
        self.runtimes["external"] = "external-endpoint", "A100"
        self.assertEqual(self.ui.fast_status()["cloud_status"], "unknown")
        C.sessions.assert_not_called()
        C.usage.assert_not_called()
        slow = self.ui.refresh_colab_cache()
        self.assertEqual(slow["active_session"]["hardware"], "A100")
        self.assertEqual(slow["active_session"]["presence"], "present")
        self.assertTrue(slow["active_session"]["reusable"])
        self.assertFalse(slow["active_session"]["owned"])
        self.assertIsNone(slow["session"])
        self.assertEqual(self.ui.fast_status()["sessions"], slow["sessions"])
        C.sessions.assert_called_once()
        C.usage.assert_called_once()

        C.sessions.side_effect = OSError("fixture network outage")
        self.wall += 30
        unknown = self.ui.refresh_colab_cache(force=True)
        self.assertTrue(unknown["sessions_failed"])
        self.assertEqual(unknown["active_session"]["presence"], "unknown")
        self.assertFalse(unknown["active_session"]["reusable"])
        self.assertEqual(unknown["active_session"]["id"], slow["active_session"]["id"])
        self.assertEqual(unknown["confirmed_at"], slow["confirmed_at"])
        self.assertEqual(unknown["checked_at"], self.wall)
        self.assertEqual(unknown["cu"], 27.0)

        self.runtimes.clear()
        C.sessions.side_effect = self.listing
        gone = self.ui.refresh_colab_cache(force=True)
        self.assertEqual(gone["cloud_status"], "confirmed")
        self.assertIsNone(gone["active_session"])
        self.assertEqual(gone["sessions"], [])
        C.colab.assert_not_called()

    def test_live_and_local_states_and_unnamed_assignments_are_distinct(self):
        self.runtimes.update({
            "idle": ("idle-endpoint", "L4"),
            "busy": ("busy-endpoint", "A100"),
            "release": ("release-endpoint", "L4"),
            "?": ("unnamed-endpoint", "H100"),
        })
        self.lease("idle", "idle-endpoint", expires=self.wall + 70)
        self.lease("busy", "busy-endpoint", expires=self.wall + 900, active_run_id="fixture-run")
        self.lease("release", "release-endpoint", expires=self.wall + 20, release_pending=True)
        self.lease("gone", "gone-endpoint", expires=self.wall + 500)
        self.lease("allocating", None, owner="pending", allocation_id="new-run")
        slow = self.ui.refresh_colab_cache()
        self.assertEqual(slow["unnamed_sessions"], 1)
        rows = {row["name"]: row for row in slow["sessions"]}
        self.assertEqual(rows["idle"]["local_state"], "idle")
        self.assertEqual(rows["idle"]["remaining_sec"], 70)
        self.assertEqual(rows["busy"]["local_state"], "busy")
        self.assertEqual(rows["release"]["local_state"], "release_pending")
        self.assertFalse(rows["busy"]["reusable"])
        self.assertFalse(rows["release"]["reusable"])
        self.assertEqual(rows["gone"]["presence"], "gone")
        self.assertEqual(rows["allocating"]["local_state"], "allocating")
        self.assertEqual(rows["allocating"]["presence"], "unknown")
        self.assertNotIn("?", rows)
        self.wall += 5
        self.assertEqual(self.ui.fast_status()["active_session"]["remaining_sec"], 65)
        C.sessions.assert_called_once()

    def test_replacement_has_its_own_id_hardware_and_no_old_lease(self):
        self.runtimes["same"] = "old-endpoint", "L4"
        self.lease("same", "old-endpoint", release_pending=True)
        old = self.ui.refresh_colab_cache()["active_session"]
        self.runtimes["same"] = "new-endpoint", "G4"
        rows = self.ui.refresh_colab_cache(force=True)["sessions"]
        new, gone = rows
        self.assertEqual(new["presence"], "present")
        self.assertEqual(new["hardware"], "G4")
        self.assertIsNone(new["local_state"])
        self.assertFalse(new["owned"])
        self.assertFalse(new["reusable"], "runner rejects a conflicting local lease")
        self.assertEqual(gone["presence"], "gone")
        self.assertEqual(gone["id"], old["id"])
        self.assertNotEqual(new["id"], old["id"])
        with self.assertRaises(HTTPException) as stale:
            self.ui.selected_endpoint(self.account, "same", old["id"])
        self.assertEqual(stale.exception.status_code, 409)
        self.assertEqual(self.ui.selected_endpoint(self.account, "same", new["id"]), "new-endpoint")
        payload = json.dumps(self.ui.fast_status())
        for secret in ("old-endpoint", "new-endpoint", self.account):
            self.assertNotIn(secret, payload)

    def test_grant_change_hides_previous_list_cu_and_run_even_when_alias_matches(self):
        self.runtimes["same"] = "shared-endpoint", "L4"
        self.lease("same", "shared-endpoint", expires=self.wall + 50)
        self.ui.running = True
        self.ui.session_account = self.account
        self.ui.on_runner_progress({"session": "same", "stage": "running", "gpu_info": {"name": "test GPU"},
                                    "_runtime_endpoint": "shared-endpoint", "_runtime_attached": True})
        old = self.ui.refresh_colab_cache()
        self.assertTrue(old["runtime_attached"])
        self.assertEqual(old["gpu"], "test GPU")
        self.assertEqual(old["gpu_source"], "cuda")
        self.account = "another-grant"
        interim = self.ui.fast_status()
        self.assertGreater(interim["revision"], old["revision"])
        self.assertEqual(interim["sessions"], [])
        self.assertIsNone(interim["cu"])
        self.assertIsNone(interim["session"])
        self.assertIsNone(interim["gpu"])
        self.assertFalse(interim["can_cancel"])
        fresh = self.ui.refresh_colab_cache()["active_session"]
        self.assertFalse(fresh["owned"])
        self.assertNotEqual(fresh["id"], old["active_session"]["id"])
        self.assertIsNone(fresh["local_state"])

    def test_ttl_uses_monotonic_time_and_failed_cu_does_not_hide_sessions(self):
        self.runtimes["gpu"] = "endpoint", "L4"
        C.usage.side_effect = OSError("fixture CU unavailable")
        self.assertTrue(self.ui.refresh_colab_cache()["cu_failed"])
        self.wall += 1000
        self.assertEqual(self.ui.refresh_colab_cache()["active_session"]["presence"], "present")
        C.sessions.assert_called_once()
        self.mono += 31
        self.ui.refresh_colab_cache()
        self.assertEqual(C.sessions.call_count, 2)
        self.mono += 90
        self.assertEqual(self.ui.fast_status()["active_session"]["presence"], "unknown")
        self.assertEqual(C.sessions.call_count, 2)

    def test_progress_uses_its_target_and_gpu_evidence_and_keeps_current_step_when_cancelling(self):
        self.runtimes["unrelated"] = "other-endpoint", "A100"
        self.ui.refresh_colab_cache()
        self.ui.running = True
        self.ui.session_account = self.account
        self.ui.requested_gpu = "H100"
        self.ui.on_runner_progress({"stage": "booting_vm", "session": "new"})
        status = self.ui.fast_status()
        self.assertEqual(status["session"], "new")
        self.assertFalse(status["runtime_attached"])
        self.assertEqual(status["requested_gpu"], "H100")
        self.assertIsNone(status["gpu"])
        for stage, step in (("syncing_code", "boot"), ("install", "install"), ("models", "models"),
                            ("comfy", "generate"), ("running", "generate"), ("cancelling", "generate"),
                            ("syncing_outputs", "sync"), ("done_with_errors", "sync")):
            with self.subTest(stage=stage):
                self.ui.on_runner_progress({
                    "stage": stage, "_runtime_endpoint": "new-endpoint", "_runtime_attached": True,
                    "gpu_info": {"name": "NVIDIA L4"} if stage == "running" else None,
                })
                status = self.ui.fast_status()
                self.assertEqual(status["progress_step"], step)
                self.assertTrue(status["runtime_attached"])
                self.assertEqual(status["session"], "new")
        self.assertEqual(status["gpu"], "NVIDIA L4")
        self.assertEqual(status["gpu_source"], "cuda")
        self.assertEqual(status["active_session"]["name"], "unrelated")
        self.ui.running = False
        self.assertIsNone(self.ui.fast_status()["session"])

    def test_concurrent_force_shares_query_but_later_force_rechecks(self):
        entered, resume, waiting = Event(), Event(), Event()
        gate = Lock()

        def enter():
            if gate.locked():
                waiting.set()
            return gate.acquire()

        # The context manager is instrumented only to make the overlap deterministic.
        self.ui.colab_query_lock = MagicMock()
        self.ui.colab_query_lock.__enter__.side_effect = enter
        self.ui.colab_query_lock.__exit__.side_effect = lambda *_: gate.release()

        def pause(**kwargs):
            entered.set()
            self.assertTrue(resume.wait(5))
            return self.listing(**kwargs)

        C.sessions.side_effect = pause
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.ui.refresh_colab_cache, True)
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.ui.refresh_colab_cache, True)
                self.assertTrue(waiting.wait(5))
                self.assertEqual(self.ui.fast_status()["cloud_status"], "unknown")
            finally:
                resume.set()
            self.assertEqual(first.result(5)["cloud_status"], "confirmed")
            self.assertEqual(second.result(5)["cloud_status"], "confirmed")
        C.sessions.assert_called_once()
        C.usage.assert_called_once()
        self.ui.refresh_colab_cache(force=True)
        self.assertEqual(C.sessions.call_count, 2)

    def test_old_query_cannot_publish_after_invalidation_or_account_round_trip(self):
        for change in ("shutdown", "account"):
            with self.subTest(change=change):
                entered, resume = Event(), Event()
                self.runtimes = {"old": ("old-endpoint", "L4")}
                old = self.ui.refresh_colab_cache(force=True)

                def pause(**kwargs):
                    snapshot = self.listing(**kwargs)
                    entered.set()
                    self.assertTrue(resume.wait(5))
                    return snapshot

                C.sessions.side_effect = pause
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(self.ui.refresh_colab_cache, True)
                    try:
                        self.assertTrue(entered.wait(5))
                        if change == "account":
                            prior = self.account
                            self.account = "other-grant"
                            self.assertEqual(self.ui.fast_status()["sessions"], [])
                            self.account = prior
                        else:
                            with self.ui.lock:
                                self.ui.invalidate_cloud()
                        self.runtimes.clear()
                    finally:
                        resume.set()
                    stale = pending.result(5)
                self.assertGreater(stale["revision"], old["revision"])
                self.assertEqual(stale["cloud_status"], "unknown")
                C.sessions.side_effect = self.listing
                self.assertEqual(self.ui.refresh_colab_cache(force=True)["sessions"], [])
