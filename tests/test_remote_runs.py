"""Exercise the transmitted run protocol with independent local launchers and tiny workers."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cstudio import colab_client as C
from remote.job_paths import bundle_digest
from remote.status import runtime_status


FIRST = "123456789abc"
SECOND = "abcdef123456"
WORKER = """\
import json, os, time
from pathlib import Path

job = Path(os.environ['CSTUDIO_DIR'])
run_id = os.environ['CSTUDIO_RUN_ID']
manifest = json.loads((job / 'manifest.json').read_text())
assert manifest['_run_id'] == run_id
with (Path(os.environ['CSTUDIO_RUNTIME_DIR']) / 'launches.txt').open('a') as fp:
    fp.write(run_id + '\\n')
status = {'run_id': run_id, 'bundle_hash': os.environ['CSTUDIO_BUNDLE_HASH'],
          'jobs': {}, 'stage': 'running', 'done': False, 'cancel_protocol': 1}

def save():
    stage = job / 'status.tmp'
    stage.write_text(json.dumps(status))
    stage.replace(job / 'status.json')

save()
marker = job / 'cancel_requests' / run_id
end = time.monotonic() + 10
while time.monotonic() < end and not (job / 'release').exists() and not marker.exists():
    time.sleep(0.01)
if marker.exists():
    status.update(stage='cancelled', done=True)
else:
    (job / 'out' / 'same-job.png').write_text(run_id)
    status.update(stage='done', done=True, jobs={'same-job': {'status': 'done', 'file': 'out/same-job.png'}})
save()
"""


@unittest.skipUnless(os.name == "posix", "Colab worker locks use flock")
class RemoteRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "cstudio"
        self.files = {"remote_runner.py": WORKER}
        self.digest = bundle_digest(self.files)
        self.addCleanup(self.release_workers)

    def job(self, run_id):
        return self.root / "runs" / run_id

    def release_workers(self):
        for job in (self.root / "runs").glob("[0-9a-f]*"):
            if (job / "launch.json").exists():
                (job / "release").touch()
        if self.root.exists():
            deadline = time.monotonic() + 5
            while runtime_status(self.root)["alive"] and time.monotonic() < deadline:
                time.sleep(0.01)

    def program(self, label, function, *args, include_start=False):
        return C.vm_program(label, function, *args, root=str(self.root), include_start=include_start)

    def execute(self, program, *, label="PREP"):
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = "CSTUDIO_" + label + " "
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])
        self.fail(f"missing result: {result.stdout}")

    def prepare(self, run_id, *, files=True, inputs=None, prompt="fixture"):
        manifest = json.dumps({"_run_id": run_id, "jobs": [{"id": "same-job", "prompt": prompt}]})
        return self.program("PREP", "prepare_run", self.digest, list(self.files), self.files if files else None,
                            manifest, inputs or {}, include_start=True)

    def launch(self, run_id):
        code = self.program("PREP", "launch_run", run_id, self.digest, list(self.files), include_start=True)
        return self.execute(code)

    def cancel(self, run_id, allow_start=False):
        return self.execute(self.program("CANCEL", "request_run_cancel", run_id, allow_start), label="CANCEL")

    def wait_until(self, run_id, *, done):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            st = runtime_status(self.root, run_id)
            if (st["done"] and not st["alive"]) if done else st["stage"] == "running":
                return st
            time.sleep(0.01)
        self.fail(f"worker state did not settle for {run_id}")

    def test_competing_launchers_and_replay_keep_previous_results(self):
        legacy = self.root / "out" / "same-job.png"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"legacy output")
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(self.execute, [self.prepare(FIRST), self.prepare(SECOND)]))
        self.assertEqual({reply["state"] for reply in replies}, {"running", "busy"})
        winner = next(reply["run_id"] for reply in replies if reply["state"] == "running")
        loser = SECOND if winner == FIRST else FIRST
        self.assertFalse(self.job(loser).exists())
        self.assertEqual(self.execute(self.prepare(winner, files=False))["state"], "running")
        self.assertEqual(runtime_status(self.root)["active_run_id"], winner)
        self.wait_until(winner, done=False)

        (self.job(winner) / "release").touch()
        self.wait_until(winner, done=True)
        self.assertEqual(self.execute(self.prepare(loser, files=False))["state"], "running")
        self.assertEqual(self.execute(self.prepare(winner, files=False))["state"], "finished")
        previous = self.execute(self.program("STATUS", "runtime_status", winner), label="STATUS")
        self.assertEqual(previous["active_run_id"], loser)
        self.assertEqual(previous["jobs"]["same-job"]["file"], "out/same-job.png")
        self.assertEqual((self.job(winner) / "out/same-job.png").read_text(), winner)
        self.wait_until(loser, done=False)
        self.assertEqual((self.root / "launches.txt").read_text().splitlines(), [winner, loser])
        self.assertEqual(legacy.read_bytes(), b"legacy output")

    def test_separate_runtime_can_run_the_same_id_concurrently(self):
        self.assertEqual(self.execute(self.prepare(FIRST))["state"], "running")
        other = self.root.with_name("other-cstudio")
        self.addCleanup(self.release_other, other, FIRST)
        manifest = json.dumps({"_run_id": FIRST, "jobs": []})
        code = C.vm_program("PREP", "prepare_run", self.digest, list(self.files), self.files, manifest, {},
                            root=str(other), include_start=True)
        self.assertEqual(self.execute(code)["state"], "running")
        self.assertTrue(runtime_status(other, FIRST)["alive"])
        self.assertTrue(runtime_status(self.root, FIRST)["alive"])

    def release_other(self, root, run_id):
        release = root / "runs" / run_id / "release"
        if release.parent.is_dir():
            release.touch()
            deadline = time.monotonic() + 4
            while runtime_status(root)["alive"] and time.monotonic() < deadline:
                time.sleep(0.01)

    def test_inputs_are_checked_on_launch_and_are_immune_to_late_upload(self):
        name = "nested/reference.png"
        data = b"valid reference"
        inputs = {name: hashlib.sha256(data).hexdigest()}
        self.assertEqual(self.execute(self.prepare(FIRST, inputs=inputs))["missing_inputs"], [name])
        upload = self.job(FIRST) / "uploads" / name
        upload.write_bytes(b"wrong reference")
        self.assertEqual(self.launch(FIRST)["missing_inputs"], [name])
        self.assertFalse((self.job(FIRST) / "launch.json").exists())
        upload.write_bytes(data)
        self.assertEqual(self.execute(self.prepare(FIRST, files=False, inputs=inputs))["state"], "prepared")
        self.assertEqual(self.launch(FIRST)["state"], "running")
        upload.write_bytes(b"a late transfer replay")
        self.assertEqual((self.job(FIRST) / "inputs" / name).read_bytes(), data)
        self.assertEqual(self.launch(FIRST)["state"], "running")

    def test_bundle_hint_is_verified_and_same_run_request_cannot_be_rebound(self):
        inputs = {"ref.png": "1" * 64}
        self.assertEqual(self.execute(self.prepare(FIRST, files=False, inputs=inputs))["state"], "bundle_required")
        self.assertEqual(self.execute(self.prepare(FIRST, inputs=inputs))["bundle_hash"], self.digest)
        manifest = self.job(FIRST) / "manifest.json"
        original = manifest.read_bytes()
        different = self.prepare(FIRST, inputs=inputs, prompt="different request")
        rejected = subprocess.run([sys.executable, "-c", different], capture_output=True, text=True, timeout=5)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("不同的遠端請求", rejected.stderr)
        self.assertEqual(manifest.read_bytes(), original)
        (self.root / "bundles" / self.digest / "remote_runner.py").write_text("corrupted fixture")
        self.assertEqual(self.execute(self.prepare(SECOND, files=False, inputs=inputs))["state"], "bundle_required")
        self.assertEqual(self.execute(self.prepare(SECOND, inputs=inputs))["state"], "prepared")
        self.assertEqual((self.root / "bundles" / self.digest / "remote_runner.py").read_text(), WORKER)

    def test_cancel_before_prepare_terminalizes_only_that_run(self):
        self.assertEqual(self.cancel(FIRST, allow_start=True), {"run_id": FIRST, "state": "pending"})
        self.assertEqual(self.execute(self.prepare(FIRST))["state"], "finished")
        self.assertEqual(runtime_status(self.root, FIRST)["stage"], "cancelled")
        self.assertFalse((self.job(FIRST) / "launch.json").exists())
        self.assertEqual(self.execute(self.prepare(SECOND, files=False))["state"], "running")
        self.assertEqual(self.cancel(FIRST)["state"], "confirmed")
        self.assertTrue(runtime_status(self.root, SECOND)["alive"])
        self.assertEqual(self.cancel(SECOND)["state"], "pending")
        self.assertEqual(self.wait_until(SECOND, done=True)["stage"], "cancelled")

    def test_prepared_run_can_be_cancelled_without_starting_a_worker(self):
        inputs = {"ref.png": "2" * 64}
        self.assertEqual(self.execute(self.prepare(FIRST, inputs=inputs))["state"], "prepared")
        self.assertEqual(self.cancel(FIRST)["state"], "confirmed")
        self.assertEqual(self.launch(FIRST)["state"], "finished")
        self.assertEqual(runtime_status(self.root, FIRST)["stage"], "cancelled")
        self.assertFalse((self.root / "launches.txt").exists())

    def test_exited_worker_does_not_relaunch_the_same_request(self):
        self.files = {"remote_runner.py": 'print("fixture exits during startup")\nraise SystemExit(1)\n'}
        self.digest = bundle_digest(self.files)
        self.assertEqual(self.execute(self.prepare(FIRST))["state"], "running")
        outcome = self.wait_until(FIRST, done=True)
        self.assertEqual(outcome["stage"], "crashed")
        self.assertIn("fixture exits during startup", outcome["error"])
        self.assertEqual(self.execute(self.prepare(FIRST))["state"], "finished")
        self.assertEqual(self.execute(self.prepare(SECOND, files=False))["state"], "running")

    def test_legacy_worker_blocks_new_launch_and_legacy_files_stay_intact(self):
        self.root.mkdir()
        legacy = self.root / "remote_runner.py"
        legacy.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
        (self.root / "manifest.json").write_text("legacy manifest", encoding="utf-8")
        with subprocess.Popen([sys.executable, str(legacy)]) as child:
            try:
                self.assertTrue(runtime_status(self.root)["alive"])
                self.assertEqual(self.execute(self.prepare(FIRST))["state"], "busy")
                self.assertFalse(self.job(FIRST).exists())
                self.assertEqual((self.root / "manifest.json").read_text(encoding="utf-8"), "legacy manifest")
            finally:
                child.terminate()
                child.wait(timeout=5)
