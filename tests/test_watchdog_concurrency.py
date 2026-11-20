"""Offline concurrency checks for the watchdog and its lease records."""
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from cstudio import colab_client as C


class WatchdogConcurrencyTests(unittest.TestCase):
    session = "gpu"

    def setUp(self):
        self.keep = Path(self.enterContext(tempfile.TemporaryDirectory())) / "keep"
        self.enterContext(patch.object(C, "KEEP", self.keep))

    def test_parallel_starters_spawn_once_and_ignore_a_reused_pid(self):
        C.set_keep(self.session, 2_000, pid=os.getpid(), owner="owned")
        original = C.keep_info(self.session)
        barrier = Barrier(2)

        def start():
            barrier.wait(timeout=5)
            C.start_watchdog(self.session)

        with patch.object(C.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)) as launch:
            with ThreadPoolExecutor(max_workers=2) as pool:
                workers = [pool.submit(start) for _ in range(2)]
                for worker in workers:
                    worker.result(timeout=6)
        launch.assert_called_once()
        record = C.keep_info(self.session)
        self.assertEqual(record["expires"], original["expires"])
        self.assertEqual(record["watchdog_pid"], 1234)
        self.assertTrue(C.same_lease(record, original))

    def test_abandoned_start_ticket_can_be_replaced_and_only_current_child_enters(self):
        C.set_keep(self.session, 2_000, owner="owned")
        with patch.object(C.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)) as launch:
            with patch.object(C.time, "time", return_value=1_000):
                C.start_watchdog(self.session)
            old_ticket = C.keep_info(self.session)["watchdog_ticket"]
            with patch.object(C.time, "time", return_value=1_000 + C.WATCHDOG_START_TIMEOUT + 1):
                C.start_watchdog(self.session)
        self.assertEqual(launch.call_count, 2)
        new_ticket = C.keep_info(self.session)["watchdog_ticket"]
        self.assertNotEqual(old_ticket, new_ticket)
        with patch.object(C, "_monitor_keep") as monitor:
            C.watchdog_loop(self.session, old_ticket)
            monitor.assert_not_called()
            C.watchdog_loop(self.session, new_ticket)
            monitor.assert_called_once()
        self.assertNotIn("watchdog_ticket", C.keep_info(self.session))

    def test_watchdog_lock_is_visible_to_another_process(self):
        C.set_keep(self.session, 2_000, owner="owned")
        helper = (
            "import sys; from pathlib import Path; from cstudio import colab_client as C; "
            "C.KEEP = Path(sys.argv[1]); print(C.watchdog_alive(sys.argv[2]))"
        )
        command = [sys.executable, "-c", helper, str(self.keep), self.session]
        with C._session_lock(self.session, "watchdog"):
            child = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
            self.assertEqual(child.stdout.strip(), "True")
            with patch.object(C.subprocess, "Popen") as launch:
                C.start_watchdog(self.session)
            launch.assert_not_called()
        child = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        self.assertEqual(child.stdout.strip(), "False")

    def test_lease_generation_guards_against_replaced_identical_record(self):
        original = C.set_keep(self.session, 2_000, owner="owned")
        self.assertTrue(C.drop_keep(self.session, expected=original))
        replacement = C.set_keep(self.session, 2_000, owner="owned")
        self.assertFalse(C.drop_keep(self.session, expected=original))
        with self.assertRaises(C.KeepChanged):
            C.set_keep(self.session, 3_000, expected=original)
        self.assertEqual(C.keep_info(self.session), replacement)

    def test_failed_launch_restores_startup_metadata_and_allows_retry(self):
        original = C.set_keep(self.session, 2_000, owner="owned")
        with patch.object(C.subprocess, "Popen", side_effect=OSError("fixture spawn failed")):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                C.start_watchdog(self.session)
        self.assertEqual(C.keep_info(self.session), original)
        with patch.object(C.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)) as launch:
            C.start_watchdog(self.session)
        launch.assert_called_once()

    def test_unconfirmed_worker_start_is_bounded_by_the_run_budget(self):
        C.set_keep(self.session, 2_000, owner="pending")
        now = 10.0

        def sleep(seconds):
            nonlocal now
            now += seconds

        clock = SimpleNamespace(time=lambda: now, monotonic=lambda: now, sleep=sleep)
        with patch.object(C, "time", clock), patch.object(C.subprocess, "Popen", return_value=SimpleNamespace(pid=1234)):
            with self.assertRaisesRegex(OSError, "watchdog 已啟動"):
                C.start_watchdog(self.session, require_ready=True, deadline=11)
        self.assertEqual(now, 11)

    def test_child_acknowledges_start_and_crash_releases_lock(self):
        C.set_keep(self.session, 2_000, owner="pending")
        real_popen = subprocess.Popen
        child = None
        helper = (
            "import sys\nfrom pathlib import Path\nfrom cstudio import colab_client as C\n"
            "C.KEEP = Path(sys.argv[1])\nC._monitor_keep = lambda *_: sys.stdin.read()\n"
            "C.watchdog_loop(sys.argv[2], sys.argv[3])"
        )

        def launch(command, **_options):
            nonlocal child
            child = real_popen([sys.executable, "-c", helper, str(self.keep), self.session, command[-1]],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return child

        try:
            with patch.object(C.subprocess, "Popen", side_effect=launch):
                C.start_watchdog(self.session, require_ready=True, deadline=C.time.monotonic() + 5)
            self.assertTrue(C.watchdog_alive(self.session))
            self.assertEqual(C.keep_info(self.session)["watchdog_pid"], child.pid)
        finally:
            if child:
                child.kill()
                child.communicate(timeout=5)
        self.assertFalse(C.watchdog_alive(self.session))


if __name__ == "__main__":
    unittest.main()
