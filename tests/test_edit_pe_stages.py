import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from remote import remote_runner as worker
from remote.models import qwen_image_21 as model


RUN = "03b000000121"
REFERENCES = ["ref-1.png", "ref-2.png", "ref-3.png"]
JOB = {"id": "edit", "mode": "edit", "inputs": REFERENCES, "refine_prompt": True,
       "prompt": "Arrange <image1>, <image2>, and <image3> in order", "timeout": 480}
ANSWER = "Place <image1>, <image2>, and <image3> from left to right."


class EditPEGraphTests(unittest.TestCase):
    def test_json_metadata_does_not_reach_image_conditioning(self):
        raw = json.dumps({"rewritten_prompt": ANSWER, "wh_ratio": "2:3", "ratio_follow": "<image10>"})
        self.assertEqual(model.normalize_edit_pe_answer(raw, JOB["prompt"], 3), ANSWER)

    def test_rewritten_prompt_remains_authoritative_when_output_is_present(self):
        raw = json.dumps({"rewritten_prompt": ANSWER, "output": "Contradictory <image4>"})
        self.assertEqual(model.normalize_edit_pe_answer(raw, JOB["prompt"], 3), ANSWER)
        for preferred, message in ((None, "rewritten_prompt"), ("Only <image1>", "缺少")):
            raw = json.dumps({"rewritten_prompt": preferred, "output": ANSWER})
            with self.subTest(preferred=preferred), self.assertRaisesRegex(ValueError, message):
                model.normalize_edit_pe_answer(raw, JOB["prompt"], 3)

    def test_missing_unexpected_and_invalid_answers_are_rejected(self):
        cases = [
            (None, "空"),
            ('{"rewritten_prompt":', "JSON"),
            ('{"ratio_follow":"<image1>"}', "rewritten_prompt"),
            ('{"output":"x","grid":3x3}', "JSON"),
            (json.dumps({"output": [ANSWER]}), "output"),
            ('{"output":" "}', "output"),
            ('{"output":"Only <image1>","ratio_follow":"<image2> <image3>"}', "缺少"),
            (json.dumps({"output": ANSWER + " From <image4>."}), "未提供"),
            ("Only <image1>", "缺少"),
            (ANSWER + " From <image4>.", "未提供"),
            (ANSWER + " From <image03>.", "未提供"),
        ]
        for raw, reason in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, reason):
                model.normalize_edit_pe_answer(raw, JOB["prompt"], 3)

    def test_reference_subset_preserves_user_intent(self):
        self.assertEqual(model.normalize_edit_pe_answer("Use <image2>", "Retain <image2>", 3), "Use <image2>")

    def test_pe_graph_has_no_image_renderer_and_input_order_matches(self):
        img = model.build_qwen21_graph(JOB)
        pe = model.build_edit_pe_graph(img, f"cstudio_{RUN}_edit")
        self.assertEqual(pe["33"]["inputs"]["source"], ["32", 0])
        self.assertEqual([pe[str(20 + i)]["inputs"]["image"] for i in range(3)], REFERENCES)
        self.assertTrue({"10", "13", "40", "54", "60", "61"}.isdisjoint(pe))


class EditPEWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "inputs").mkdir()
        self.shas = {}
        for name in REFERENCES:
            content = name.encode()
            (self.tmp / "inputs" / name).write_bytes(content)
            self.shas[name] = hashlib.sha256(content).hexdigest()
        self.comfy = self.tmp / "ComfyUI"
        self.comfy.mkdir()
        self.state = {"jobs": {}, "bundle_hash": "a" * 64, "log": [],
                      "environment": {}, "gpu_info": {"name": "L4"}}
        self.now = 0
        self.inflight = {}
        self.fitted = {}
        self.collected = Mock(side_effect=self.collect)
        self.submissions = []
        self.timeouts = []
        self.enterContext(patch.multiple(
            worker, JOB=self.tmp, OUT=self.tmp / "out", WORK=self.tmp / "work", COMFY=self.comfy, MANAGED_RUN_ID=RUN,
            STATE=self.state, IN_FLIGHT=self.inflight, FITTED=self.fitted, save=Mock(),
            install=Mock(return_value="fixture"), download_models=Mock(side_effect=self.download),
            start_comfy=Mock(return_value={"args": ["main.py"], "verified": True, "versions": {}}),
            collect_output_image=self.collected,
        ))
        self.enterContext(patch.object(worker.time, "monotonic", side_effect=lambda: self.now))

    @staticmethod
    def collect(_, path, **kwargs):
        path.write_bytes(b"fixture PNG")

    @staticmethod
    def download(models, **kwargs):
        return {item: {"repo": item[0], "file": item[1]} for item in models}

    def manifest(self, jobs):
        raw = json.dumps({"_run_id": RUN, "_fingerprint_version": 2, "jobs": jobs}).encode()
        (self.tmp / "manifest.json").write_bytes(raw)
        (self.tmp / "request.json").write_text(json.dumps({
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "bundle_hash": "a" * 64, "input_shas": self.shas,
        }))

    def submit(self, graph, *, pin_models, deadline):
        self.assertTrue(pin_models)
        self.submissions.append(graph)
        self.assertGreater(deadline, self.now)
        pid = f"prompt-{len(self.submissions)}"
        self.inflight[pid] = True
        return pid

    def run_jobs(self, jobs, wait):
        self.manifest(jobs)
        with patch.multiple(worker, queue_prompt=Mock(side_effect=self.submit), wait_history=Mock(side_effect=wait)):
            with redirect_stdout(StringIO()):
                worker.run_batch()

    def test_refined_answer_and_both_prompt_receipts_are_published(self):
        raw = json.dumps({"rewritten_prompt": ANSWER, "ratio_follow": "<image1>"})

        def wait(pid, *, timeout, what):
            self.timeouts.append(timeout)
            self.inflight.pop(pid)
            self.now += 120
            return {"outputs": {"33": {"text": [raw]}}} if what.endswith(" PE") else {}

        self.run_jobs([JOB], wait)
        self.assertEqual(self.timeouts, [480, 360])
        pe, img = self.submissions
        self.assertNotIn("61", pe)
        self.assertNotIn("32", img)
        self.assertEqual(img["40"]["inputs"]["prompt"], ANSWER)
        self.assertEqual([img[str(20 + i)]["inputs"]["image"] for i in range(3)],
                         [f"cstudio_{RUN}_edit_in{i}.png" for i in range(1, 4)])
        rec = self.state["jobs"]["edit"]
        self.assertEqual(rec["status"], "done")
        self.assertEqual(rec["refined_prompt"], ANSWER)
        self.assertEqual(rec["execution"]["pe"], {
            "prompt_id": "prompt-1", "image_prompt_id": "prompt-2",
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(ANSWER.encode()).hexdigest(),
        })
        self.collected.assert_called_once()

    def test_output_json_reaches_image_encoder_and_receipts(self):
        raw = json.dumps({"output": ANSWER})

        def wait(pid, *, timeout, what):
            self.inflight.pop(pid)
            return {"outputs": {"33": {"text": [raw]}}} if what.endswith(" PE") else {}

        self.run_jobs([JOB], wait)
        _, image = self.submissions
        self.assertEqual(image["40"]["inputs"]["prompt"], ANSWER)
        receipt = self.state["jobs"]["edit"]
        self.assertEqual(receipt["status"], "done")
        self.assertEqual(receipt["refined_prompt"], ANSWER)
        self.assertEqual(receipt["execution"]["pe"]["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(receipt["execution"]["pe"]["answer_sha256"], hashlib.sha256(ANSWER.encode()).hexdigest())

    def test_invalid_pe_has_no_render_or_output(self):
        def wait(pid, **kwargs):
            self.inflight.pop(pid)
            return {"outputs": {"33": {"text": ['{"rewritten_prompt":"x","grid":3x3}']}}}

        self.run_jobs([JOB], wait)
        self.assertEqual(len(self.submissions), 1)
        self.assertIn("JSON", self.state["jobs"]["edit"]["error"])
        self.collected.assert_not_called()

    def test_confirmed_abort_of_timed_out_pe_precedes_next_submit(self):
        cancelled = []

        def wait(pid, **kwargs):
            if pid == "prompt-1":
                raise RuntimeError("fixture PE deadline")
            self.assertTrue(cancelled)
            self.inflight.pop(pid)
            return {"outputs": {"33": {"text": [ANSWER]}}}

        def cancel():
            cancelled.extend(self.inflight)
            self.inflight.clear()

        with patch.object(worker, "cancel_prompts", side_effect=cancel):
            self.run_jobs([JOB, {**JOB, "id": "next"}], wait)
        self.assertEqual(cancelled, ["prompt-1"])
        self.assertEqual(self.state["jobs"]["edit"]["status"], "error")
        self.assertEqual(self.state["jobs"]["next"]["status"], "done")

    def test_unknown_cancel_blocks_next_job(self):
        with patch.object(worker, "cancel_prompts", side_effect=worker.CancellationUnconfirmed("fixture")):
            with self.assertRaises(worker.CancellationUnconfirmed):
                self.run_jobs([JOB, {**JOB, "id": "next"}], Mock(side_effect=RuntimeError("fixture PE deadline")))
        self.assertEqual(len(self.submissions), 1)
        self.collected.assert_not_called()


if __name__ == "__main__":
    unittest.main()
