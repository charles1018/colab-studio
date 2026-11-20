"""Offline manifest checks across CLI, schema and the Colab worker."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator
from PIL import Image

from cstudio import colab_client as C, runner as R
from remote import manifest as M, remote_runner as worker


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.base / "manifest.json"
        self.job = {"id": "sample", "prompt": "A tree"}

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_demo_and_aliases_preserve_overrides_and_advanced_graph_inputs(self):
        demo = R.ROOT / "jobs" / "demo_qwen21" / "manifest.json"
        self.assertEqual(R.load_manifest(demo)[0], json.loads(demo.read_text()))
        data = {
            "model": " Qwen21 ", "quant": " bF16 ", "refine_prompt": True, "thinking": False,
            "pe_max_length": 32768, "cache_device": " CPU ", "cache_dtype": "int4",
            "text_encoder_variant": "W4A8", "resolution": " 4mp ", "jobs": [
                {**self.job, "width": 549, "height": 1024, "pe_max_length": 512,
                 "cfg": 2.5, "sampler": " dpmpp_2m ", "sigmas": "[1., .75 .5, 0]", "timeout": 900,
                 "seed": M.MAX_SEED, "negative_prompt": "", "transparent": True},
                {"id": "second", "prompt": "A house", "quant": "int8", "refine_prompt": False, "resolution": "1MP"},
            ],
        }
        self.write(data)
        normal, base = R.load_manifest(self.path, cli_quant=" q4_k_m ", exit_on_error=False)
        self.assertEqual(base, self.base)
        self.assertEqual(normal["model"], "qwen21")
        first, second = normal["jobs"]
        self.assertEqual(normal["quant"], "BF16")
        self.assertEqual(normal["resolution"], "2K")
        self.assertEqual(second["resolution"], "1K")
        q = R.get_model_plugin(normal["model"])
        graph = q.build_qwen21_graph(first, normal)
        self.assertEqual(graph["10"]["class_type"], "UNETLoader")
        self.assertEqual(graph["11"]["inputs"]["device"], "cpu")
        self.assertEqual(graph["11"]["inputs"]["dtype"], "int4")
        self.assertEqual(graph["32"]["inputs"]["max_length"], 512)
        self.assertFalse(graph["32"]["inputs"]["thinking"])
        self.assertEqual(graph["41"]["inputs"]["width"], 544)
        self.assertEqual(graph["51"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(graph["52"]["inputs"]["noise_seed"], M.MAX_SEED)
        self.assertEqual(graph["53"]["inputs"]["cfg"], 2.5)
        second_graph = q.build_qwen21_graph(second, normal)
        self.assertEqual(second_graph["10"]["inputs"]["unet_name"], q.gguf_filename("INT8"))
        self.assertNotIn("32", second_graph)
        with redirect_stdout(io.StringIO()) as stdout:
            plan = R.plan_batch(self.path, cli_quant=" q4_k_m ")
        self.assertIn("quant=Q4_K_M", stdout.getvalue())
        self.assertTrue(any("Q4_K_M.gguf" in model["file"] for model in plan["models"]))
        self.assertEqual(json.loads(self.path.read_text()), data)

    def test_edit_preserves_order_and_existing_canvas_rules(self):
        names = [f"ref_{index}.png" for index in range(10)]
        for name in names:
            Image.new("RGBA", (2, 2), (10, 20, 30, 40)).save(self.base / name)
        q = R.get_model_plugin()
        for refs, settings, latent, target in (
            (names[:1], {"edit_resolution": 0}, ["40", 2], 0),
            (names[::-1], {"resolution": " 2048 "}, ["40", 2], 2048),
            (names, {"custom_size": True, "width": 4096, "height": 32}, ["41", 0], 1024),
        ):
            with self.subTest(refs=len(refs), settings=settings):
                self.write({"jobs": [{**self.job, **settings, "mode": " EDIT ", "inputs": refs}]})
                manifest, _ = R.load_manifest(self.path, exit_on_error=False)
                graph = q.build_qwen21_graph(manifest["jobs"][0], manifest)
                self.assertEqual([graph[str(20 + index)]["inputs"]["image"] for index in range(len(refs))], refs)
                self.assertEqual(graph["54"]["inputs"]["latent_image"], latent)
                self.assertEqual(graph["40"]["inputs"]["resolution"], target)

    def test_invalid_job_fields_issue_no_colab_commands(self):
        cases = {
            "prompt": (None, "    ", 42), "negative_prompt": (None, False), "mode": (None, 2),
            "refine_prompt": ("false", None, 1), "transparent": (1,), "thinking": ("true",),
            "seed": (-1, True, 1.0, "42", M.MAX_SEED + 1), "resolution": (1024, None, "3K"),
            "inputs": (None, "ref.png", [1], ["ref.png"]), "width": (True, 8192, 31, 512),
            "cfg": ("1.0", True, -0.1, 101.0, 10**400), "pe_max_length": (0, 32769, "77"),
            "timeout": (0, M.MAX_JOB_TIMEOUT + 1, None), "edit_resolution": (0,),
            "cache_device": ("disk",), "cache_dtype": ("int16",), "sampler": ("typo",),
            "sigmas": ("1", "1, NaN, 0", "1e-2, 0", "1, 0.5, 0.5", "1, -1", "1, 2", "1, 0.999999999, 0"),
            "id": ("a\n", "CON", ""), "text_encoder_variant": ("fallback",), "unknown": (True,),
        }
        with patch.multiple(C, colab=Mock(), sessions_checked=Mock(), usage=Mock(), new_runtime=Mock(),
                            token_ok=Mock()):
            for field, values in cases.items():
                for value in values:
                    with self.subTest(field=field, value=value):
                        self.write({"jobs": [{**self.job, field: value}]})
                        with self.assertRaisesRegex(RuntimeError, "manifest 驗證未通過"):
                            R.run_manifest(self.path, force=True, exit_on_error=False)
            for command in (C.colab, C.sessions_checked, C.usage, C.new_runtime, C.token_ok):
                command.assert_not_called()
        self.assertFalse((self.base / "out").exists())

    def test_invalid_top_level_and_jobs_cannot_be_hidden_by_overrides(self):
        invalid = (
            None, [], {"jobs": []}, {"jobs": "wrong"}, {"jobs": [self.job, self.job]},
            {"jobs": [self.job], "seed": 42}, {"jobs": [self.job], "_cli_quant": "BF16"},
            {"jobs": [self.job], "model": None}, {"jobs": [self.job], "quant": "typo"},
            {"jobs": [{**self.job, "quant": "typo"}], "quant": "INT8"},
            {"jobs": [{**self.job, "sampler": "ddim", "seed": M.MAX_SEED}]},
            {"jobs": [{**self.job, "mode": "edit", "inputs": ["one.png"] * 11}]},
            {"jobs": [{**self.job, "mode": "edit", "inputs": []}]},
            {"jobs": [{**self.job, "mode": "edit", "inputs": ["ref.png"], "edit_resolution": -1}]},
            {"jobs": [{**self.job, "mode": "edit", "inputs": ["ref.png"], "custom_size": False, "aspect_ratio": "1:1"}]},
        )
        for data in invalid:
            with self.subTest(data=data):
                self.write(data)
                with self.assertRaisesRegex(RuntimeError, "manifest 驗證未通過"):
                    R.load_manifest(self.path, cli_quant="bf16", exit_on_error=False)
        self.write({"jobs": [self.job]})
        with self.assertRaisesRegex(RuntimeError, "--quant"):
            R.load_manifest(self.path, cli_quant="typo", exit_on_error=False)

    def test_nonstandard_duplicate_and_overflowing_json_fails_before_run(self):
        for literal, expected in (
            ('"cfg": NaN', "數值必須有限"), ('"cfg": Infinity', "數值必須有限"),
            ('"cfg": -Infinity', "數值必須有限"), ('"cfg": 1e309', "cfg"),
            ('"seed": 1, "seed": 2', "欄位重複"),
        ):
            with self.subTest(literal=literal):
                self.path.write_text('{"jobs": [{"id": "job", "prompt": "fixture", ' + literal + '}]}')
                with self.assertRaisesRegex(RuntimeError, expected):
                    R.load_manifest(self.path, exit_on_error=False)

    def test_public_schema_agrees_on_supported_fields_and_type_constraints(self):
        schema_path = R.ROOT / "schema" / "manifest.schema.json"
        schema = json.loads(schema_path.read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        defs = schema["$defs"]
        shared = defs["shared"]["properties"].keys()
        self.assertEqual(set(M.MANIFEST_FIELDS) | {"jobs"}, set(schema["properties"]) | shared)
        self.assertEqual(set(M.JOB_FIELDS), set(defs["job"]["properties"]) | shared)
        for valid, job_fields in (
            (True, {"quant": " bf16 ", "resolution": "2048", "cfg": 0, "sigmas": "1 0.25 0"}),
            (True, {"mode": " EDIT ", "inputs": ["nested/a.png"], "edit_resolution": 4096}),
            (True, {"width": 32, "height": 4096, "seed": M.MAX_SEED}),
            (False, {"mode": "edit", "inputs": []}), (False, {"mode": "t2i", "inputs": ["nested/a.png"]}),
            (False, {"custom_size": False}), (False, {"width": 512}), (False, {"seed": M.MAX_SEED + 1}),
            (False, {"sampler": "ddim", "seed": M.MAX_SEED}), (False, {"cfg": 101}),
            (False, {"thinking": "false"}), (False, {"pe_max_length": 32769}), (False, {"timeout": 0}),
        ):
            with self.subTest(job_fields=job_fields):
                data = {"model": "QWEN21", "jobs": [{**self.job, **job_fields}]}
                self.assertEqual(validator.is_valid(data), valid)
                self.assertEqual(not M.normalize_manifest(data, None)[1], valid)

    def test_worker_checks_manifest_and_wire_types_before_setup(self):
        state = {"log": [], "jobs": {}}
        with patch.multiple(worker, JOB=self.base, STATUS=self.base / "status.json", STATE=state,
                            MANAGED_RUN_ID=None, install=Mock(), download_models=Mock(), start_comfy=Mock()):
            for data in (
                {"jobs": [{**self.job, "seed": -1}]},
                {"jobs": [{**self.job, "custom_size": "false"}]},
                {"jobs": [self.job], "_pin_versions": "false"},
                {"jobs": [self.job], "_fingerprint_version": True},
                {"jobs": [self.job], "_cli_quant": "invalid"},
            ):
                with self.subTest(data=data):
                    self.write(data)
                    with self.assertRaises(ValueError):
                        worker.run_batch()
            worker.install.assert_not_called()
            worker.download_models.assert_not_called()
            worker.start_comfy.assert_not_called()
