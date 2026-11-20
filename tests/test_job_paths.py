"""Offline checks for the path rules used by the CLI and Colab runner."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cstudio import runner as R
from remote import job_paths
from remote import remote_runner as remote


class JobPathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.job = self.root / "job"
        self.job.mkdir()

    def link(self, path, target, *, directory=False):
        try:
            path.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links unavailable")

    def write_manifest(self, inputs):
        path = self.job / "manifest.json"
        path.write_text(json.dumps({
            "jobs": [{"id": "edit", "mode": "edit", "prompt": "fixture", "inputs": inputs}],
        }), encoding="utf-8")
        return path

    def test_cli_keeps_flat_and_nested_manifest_inputs(self):
        (self.job / "inputs").mkdir()
        (self.job / "flat.png").write_bytes(b"fixture flat")
        (self.job / "inputs" / "參考 圖.png").write_bytes(b"fixture nested")
        manifest = self.write_manifest(["flat.png", "inputs/參考 圖.png"])
        with redirect_stdout(io.StringIO()):
            result = R.plan_batch(manifest)
        self.assertEqual(result["todo"], ["edit"])
        loaded, base = R.load_manifest(manifest, exit_on_error=False)
        self.assertEqual(base, self.job)
        self.assertEqual(loaded["jobs"][0]["inputs"], ["flat.png", "inputs/參考 圖.png"])

    def test_manifest_rejects_noncanonical_paths_and_nonfiles(self):
        sibling = self.root / "sibling"
        sibling.mkdir()
        (sibling / "fixture.png").write_bytes(b"fixture")
        (self.job / "folder.png").mkdir()
        for ref in (
            "../sibling/fixture.png", str(sibling / "fixture.png"), "C:/fixture.png",
            r"inputs\fixture.png", "nested//fixture.png", "nested/./fixture.png",
            "nested/.. /fixture.png", "fixture.png:other", "COM1.png", "ref\n.png",
            1, "folder.png",
        ):
            with self.subTest(ref=ref), self.assertRaisesRegex(RuntimeError, "manifest 驗證未通過"):
                R.load_manifest(self.write_manifest([ref]), exit_on_error=False)

    def test_reference_paths_do_not_follow_links_or_continue_after_io_errors(self):
        inside = self.job / "inside.png"
        inside.write_bytes(b"inside")
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        self.link(self.job / "internal.png", inside)
        self.link(self.job / "external.png", outside)
        self.link(self.job / "nested", self.root, directory=True)
        self.link(self.job / "dangling.png", self.root / "missing.png")
        self.link(self.job / "loop.png", "loop.png")
        for ref in ("internal.png", "external.png", "nested/outside.png", "dangling.png", "loop.png"):
            with self.subTest(ref=ref), self.assertRaisesRegex(RuntimeError, "符號連結"):
                R.load_manifest(self.write_manifest([ref]), exit_on_error=False)
        manifest = self.write_manifest(["inside.png"])
        original_lstat = Path.lstat

        def unreadable(path):
            if path == inside:
                raise PermissionError("fixture")
            return original_lstat(path)

        with patch.object(Path, "lstat", unreadable), self.assertRaisesRegex(RuntimeError, "無法使用輸入圖片"):
            R.load_manifest(manifest, exit_on_error=False)

    def test_remote_bundle_uses_same_path_checks_before_marking_uploads_ready(self):
        sha = hashlib.sha256(b"reference").hexdigest()
        runner = R.bundle_and_launch_code(None, "fixture", "{}", {"nested/ref.png": sha})
        runner = runner.replace("Path('/content/cstudio')", f"Path({str(self.job)!r})")
        compiled = compile(runner, "<fixture bundle>", "exec")
        with patch("subprocess.Popen") as spawn, redirect_stdout(io.StringIO()) as out:
            exec(compiled, {})
        spawn.assert_not_called()
        self.assertTrue((self.job / "inputs" / "nested").is_dir())
        self.assertEqual(json.loads(out.getvalue().split("CSTUDIO_PREP ", 1)[1]), {
            "missing_inputs": ["nested/ref.png"], "launched": False,
        })
        self.link(self.job / "inputs" / "nested" / "ref.png", self.root / "yard.png")
        with patch("subprocess.Popen") as spawn, self.assertRaisesRegex(ValueError, "符號連結"):
            exec(compiled, {})
        spawn.assert_not_called()

    def test_both_remote_start_paths_verify_files_before_launch(self):
        worker = self.job / "remote_runner.py"
        worker.write_text("# synthetic worker", encoding="utf-8")
        (self.job / "job_paths.py").write_text((R.REMOTE / "job_paths.py").read_text(encoding="utf-8"), encoding="utf-8")
        inline = R.bundle_and_launch_code(None, "fixture", "{}", {})
        fallback = (R.REMOTE / "start.py").read_text(encoding="utf-8")
        for source, token in ((inline, "'/content/cstudio'"), (fallback, '"/content/cstudio"')):
            with self.subTest(token=token):
                code = compile(source.replace(token, repr(str(self.job))), "<fixture start>", "exec")
                with (
                    patch.object(sys, "path", sys.path.copy()),
                    patch.dict(sys.modules, {"job_paths": job_paths}),
                    patch("subprocess.run", return_value=SimpleNamespace(stdout="")),
                    patch("subprocess.Popen", return_value=SimpleNamespace(pid=42)) as spawn,
                    redirect_stdout(io.StringIO()),
                ):
                    exec(code, {})
                spawn.assert_called_once()
                self.assertEqual(spawn.call_args.args[0], [sys.executable, str(worker)])
                self.assertEqual(spawn.call_args.kwargs["cwd"], str(self.job))

        log = self.job / "remote_runner.log"
        log.unlink()
        old_output = self.job / "out" / "old.png"
        old_output.write_bytes(b"keep prior output")
        self.link(log, old_output)
        for source, token in ((inline, "'/content/cstudio'"), (fallback, '"/content/cstudio"')):
            with self.subTest(token=token):
                code = compile(source.replace(token, repr(str(self.job))), "<fixture start>", "exec")
                with (
                    patch.object(sys, "path", sys.path.copy()),
                    patch.dict(sys.modules, {"job_paths": job_paths}),
                    patch("subprocess.run", return_value=SimpleNamespace(stdout="")),
                    patch("subprocess.Popen") as spawn,
                    self.assertRaisesRegex(ValueError, "符號連結"),
                ):
                    exec(code, {})
                spawn.assert_not_called()
                self.assertEqual(old_output.read_bytes(), b"keep prior output")

    def test_remote_staging_keeps_nested_inputs_and_confines_both_ends(self):
        comfy = self.root / "comfy"
        comfy.mkdir()
        refs = self.job / "inputs" / "nested"
        refs.mkdir(parents=True)
        (refs / "ref.png").write_bytes(b"fixture")
        with patch.multiple(remote, JOB=self.job, COMFY=comfy):
            self.assertEqual(remote.stage_input_image("nested/ref.png", "edit", 1), "cstudio_edit_in1.png")
            self.assertEqual((comfy / "input" / "cstudio_edit_in1.png").read_bytes(), b"fixture")
            outside = self.root / "outside.png"
            outside.write_bytes(b"outside")
            self.link(refs / "alias.png", outside)
            with self.assertRaisesRegex(ValueError, "符號連結"):
                remote.stage_input_image("nested/alias.png", "edit", 2)
            self.link(comfy / "input" / "cstudio_edit_in2.png", outside)
            with self.assertRaisesRegex(ValueError, "符號連結"):
                remote.stage_input_image("nested/ref.png", "edit", 2)
            self.assertEqual(outside.read_bytes(), b"outside")

    def test_comfy_history_uses_only_expected_save_image_file(self):
        comfy = self.root / "comfy"
        (comfy / "output").mkdir(parents=True)
        filename = "cstudio_edit_00001_.png"
        file_info = {"filename": filename, "subfolder": "", "type": "output"}
        dst = self.job / "edit.png"

        def collect(node, info):
            return remote.collect_output_image(
                {"outputs": {node: {"images": [info]}}}, dst, save_node="61", prefix="cstudio_edit",
            )

        with patch.object(remote, "COMFY", comfy), patch.object(remote.urllib.request, "urlopen") as fetch:
            for node, info in (
                ("32", file_info), ("61", dict(file_info, type="temp")),
                ("61", dict(file_info, subfolder="nested")), ("61", dict(file_info, filename="other.png")),
                ("61", dict(file_info, filename=f"nested/{filename}")),
            ):
                with self.subTest(node=node, info=info), self.assertRaises(RuntimeError):
                    collect(node, info)
            self.assertFalse(dst.exists())
            source = comfy / "output" / filename
            source.write_bytes(b"local result")
            self.assertEqual(collect("61", file_info), dst)
            self.assertEqual(dst.read_bytes(), b"local result")
            source.unlink()
            self.link(source, dst)
            with self.assertRaisesRegex(ValueError, "符號連結"):
                collect("61", file_info)
            fetch.assert_not_called()
            source.unlink()
            fetch.return_value = io.BytesIO(b"HTTP result")
            self.assertEqual(collect("61", file_info), dst)
            self.assertEqual(dst.read_bytes(), b"HTTP result")
            fetch.assert_called_once()
