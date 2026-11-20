"""Offline model integrity tests with synthetic weights and a Hugging Face substitute."""
import errno
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from remote import remote_runner as remote


REPO = "fixture/model"
FILE = "diffusion_models/model.safetensors"
FOLDER = "diffusion_models"
REVISION = "1" * 40
OLD = b"a" * 2048
NEW = b"b" * 2048
BAD = b"c" * 2048


def pins_for(data, revision=REVISION):
    return {"models": [{
        "repo": REPO, "file": FILE, "dest": FOLDER, "revision": revision,
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }]}


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.models = self.root / "comfy/models"
        self.dest = self.models / FILE
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(OLD)
        self.payload = NEW
        self.transfer = Mock(side_effect=self.fake_transfer)
        self.log = Mock()
        self.enterContext(patch.multiple(
            remote, COMFY=self.models.parent, HF_MODELS=self.root / "hf",
            JOB=self.root, STATE={}, setup_command=self.transfer, log=self.log,
        ))
        self.free_space = self.enterContext(patch.object(
            remote.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10),
        ))
        self.enterContext(patch.dict(os.environ))

    def fake_transfer(self, args, *, read_stdout):
        repo, name, _revision, directory = args[-4:]
        self.assertTrue(read_stdout)
        self.assertEqual((repo, name), (REPO, FILE))
        candidate = Path(directory) / name
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(self.payload)
        return SimpleNamespace(stdout="download log\nCSTUDIO_MODEL " + json.dumps(str(candidate)))

    def download(self, pins):
        return remote.download_models([(REPO, FILE, FOLDER)], pins)

    def assert_no_staging_files(self):
        self.assertEqual(list(self.models.rglob(".cstudio-model-*")), [])
        self.assertEqual(list(remote.HF_MODELS.glob(".cstudio-download-*")), [])

    def test_unverified_same_size_cache_is_replaced_with_verified_download(self):
        self.download(pins_for(NEW))
        self.assertEqual(self.dest.read_bytes(), NEW)
        self.assertEqual(self.transfer.call_args.args[0][-2], REVISION)
        with patch.object(remote, "_hash_model", side_effect=AssertionError("warm cache re-read")):
            self.download(pins_for(NEW))
        self.transfer.assert_called_once()
        self.assert_no_staging_files()

    def test_warm_cache_rechecks_mutation_even_when_mtime_is_restored(self):
        self.download(pins_for(OLD))
        self.transfer.assert_not_called()
        before = self.dest.stat()
        self.dest.write_bytes(BAD)
        os.utime(self.dest, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.payload = OLD
        self.download(pins_for(OLD))
        self.transfer.assert_called_once()
        self.assertEqual(self.dest.read_bytes(), OLD)

    def test_warm_cache_rechecks_new_inode_pin_and_missing_record(self):
        record = remote._model_verification_path(self.dest)
        self.download(pins_for(OLD))
        replacement = self.dest.with_suffix(".part")
        replacement.write_bytes(OLD)
        replacement.replace(self.dest)
        with patch.object(remote, "_hash_model", wraps=remote._hash_model) as verify:
            self.download(pins_for(OLD))
            self.download(pins_for(OLD, "2" * 40))
            record.write_text("incomplete json", encoding="utf-8")
            self.download(pins_for(OLD, "2" * 40))
            self.assertEqual(verify.call_count, 3)
        self.transfer.assert_not_called()

    def test_wrong_digest_or_size_keeps_verified_old_file_and_record(self):
        self.download(pins_for(OLD))
        record = remote._model_verification_path(self.dest)
        original_record = record.read_bytes()
        for payload, message in ((BAD, "SHA-256"), (b"short", "大小不符")):
            with self.subTest(message=message):
                self.payload = payload
                with self.assertRaisesRegex(RuntimeError, message):
                    self.download(pins_for(NEW))
                self.assertEqual(self.dest.read_bytes(), OLD)
                self.assertEqual(record.read_bytes(), original_record)
                self.assert_no_staging_files()

    def test_interrupted_transfer_keeps_old_file_and_removes_partial_download(self):
        def interrupted(*args, **kwargs):
            self.fake_transfer(*args, **kwargs)
            raise remote.BatchCancelled()

        self.transfer.side_effect = interrupted
        with self.assertRaises(remote.BatchCancelled):
            self.download(pins_for(NEW + b"more"))
        self.assertEqual(self.dest.read_bytes(), OLD)
        self.assert_no_staging_files()

    def test_insufficient_space_for_replacement_does_not_start_transfer(self):
        self.free_space.return_value.free = len(NEW) - 1
        with self.assertRaisesRegex(RuntimeError, "磁碟空間不足"):
            self.download(pins_for(NEW))
        self.transfer.assert_not_called()
        self.assertEqual(self.dest.read_bytes(), OLD)

    def test_cross_filesystem_copy_checks_staged_file_and_replaces_atomically(self):
        replace = Path.replace

        def cross_device(path, target):
            if path.is_relative_to(remote.HF_MODELS):
                raise OSError(errno.EXDEV, "different filesystem")
            if target == self.dest:
                self.assertEqual(self.dest.read_bytes(), OLD)
                self.assertEqual(path.stat().st_dev, self.dest.stat().st_dev)
                self.assertEqual(path.read_bytes(), NEW)
            return replace(path, target)

        with patch.object(Path, "replace", cross_device), \
             patch.object(remote, "_hash_model", wraps=remote._hash_model) as verify:
            self.download(pins_for(NEW))
        self.assertEqual(Path(verify.call_args.args[0].name).parent.parent, self.dest.parent)
        self.assertEqual(self.dest.read_bytes(), NEW)
        self.assert_no_staging_files()

    def test_cross_filesystem_copy_rejects_wrong_hash_and_keeps_old_on_rename_failure(self):
        replace = Path.replace

        def replace_or_fail(path, target):
            if path.is_relative_to(remote.HF_MODELS):
                raise OSError(errno.EXDEV, "different filesystem")
            if target == self.dest:
                raise PermissionError("fixture final rename failed")
            return replace(path, target)

        with patch.object(Path, "replace", replace_or_fail):
            self.payload = BAD
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.download(pins_for(NEW))
            self.assertEqual(self.dest.read_bytes(), OLD)
            self.assert_no_staging_files()
            self.payload = NEW
            with self.assertRaisesRegex(PermissionError, "final rename failed"):
                self.download(pins_for(NEW))
        self.assertEqual(self.dest.read_bytes(), OLD)
        self.assert_no_staging_files()

    def test_cross_filesystem_copy_on_local_mounts(self):
        other_fs = Path("/dev/shm")
        if not other_fs.is_dir() or other_fs.stat().st_dev == self.root.stat().st_dev:
            self.skipTest("requires two writable local filesystems")
        try:
            download_root = self.enterContext(tempfile.TemporaryDirectory(dir=other_fs))
        except PermissionError:
            self.skipTest("/dev/shm is not writable")
        with patch.object(remote, "HF_MODELS", Path(download_root)):
            self.download(pins_for(NEW))
            self.assertEqual(self.free_space.call_count, 2)
            self.assert_no_staging_files()
        self.assertEqual(self.dest.read_bytes(), NEW)

    def test_pinned_native_model_checks_legacy_unet_before_diffusion_models(self):
        self.dest.write_bytes(NEW)
        legacy = self.models / "unet" / self.dest.name
        legacy.parent.mkdir()
        legacy.write_bytes(BAD)
        provenance = self.download(pins_for(NEW))
        self.transfer.assert_called_once()
        self.assertEqual(legacy.read_bytes(), NEW)
        self.assertEqual(provenance[(REPO, FILE, FOLDER)], {
            "repo": REPO, "file": FILE, "path": f"models/unet/{legacy.name}",
            "bytes": len(NEW), "sha256": hashlib.sha256(NEW).hexdigest(), "revision": REVISION,
        })

    def test_pinned_mode_requires_a_valid_digest_for_each_requested_file(self):
        pins = pins_for(OLD)
        del pins["models"][0]["sha256"]
        for invalid_pins in ({}, pins):
            with self.subTest(pins=invalid_pins), self.assertRaisesRegex(RuntimeError, "模型釘選資料"):
                self.download(invalid_pins)
        self.transfer.assert_not_called()

    def test_unpinned_mode_reuses_existing_weights_and_downloads_default_revision(self):
        with patch.object(remote, "_hash_model", side_effect=AssertionError("unpinned mode hashed a file")):
            self.download(None)
            self.transfer.assert_not_called()
            self.dest.unlink()
            self.download(None)
        self.assertEqual(self.transfer.call_args.args[0][-2:], ["", str(remote.HF_MODELS)])
        self.assertEqual(self.dest.read_bytes(), NEW)
        self.assertFalse(remote._model_verification_path(self.dest).exists())

    def test_remote_batch_uses_pin_flags_and_selects_only_needed_weights(self):
        required = [
            ("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q4_K_M.gguf", "unet"),
            ("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
            ("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
        ]
        pins = json.loads(Path(remote.__file__).with_name("versions.json").read_text(encoding="utf-8"))
        versions = self.root / "versions.json"
        versions.write_text(json.dumps(pins), encoding="utf-8")
        manifest = self.root / "manifest.json"
        jobs = [{"id": "demo", "prompt": "fixture", "quant": "Q4_K_M"}]
        for flags, version_pins, model_pins in (
            ({}, pins, pins), ({"_pin_models": False}, pins, None),
            ({"_pin_versions": False, "_pin_models": False}, None, None),
        ):
            with self.subTest(flags=flags), patch.multiple(
                remote, save=Mock(), install=Mock(), download_models=Mock(side_effect=remote.BatchCancelled),
            ):
                manifest.write_text(json.dumps({"jobs": jobs, **flags}), encoding="utf-8")
                with self.assertRaises(remote.BatchCancelled):
                    remote.run_batch()
                remote.install.assert_called_once_with(version_pins)
                remote.download_models.assert_called_once_with(required, pins=model_pins)

        versions.unlink()
        manifest.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        with patch.multiple(remote, save=Mock(), install=Mock()):
            with self.assertRaisesRegex(RuntimeError, "versions.json"):
                remote.run_batch()
            remote.install.assert_not_called()

    def test_pinned_prompt_does_not_use_unverified_same_basename_in_subdirectory(self):
        graph = {"10": {"class_type": "UNETLoader", "inputs": {"unet_name": self.dest.name}}}
        options = {"UNETLoader": {"input": {"required": {"unet_name": [[f"other/{self.dest.name}"]]}}}}
        with patch.multiple(remote, _object_info=Mock(return_value=options), call=Mock()):
            with self.assertRaisesRegex(RuntimeError, "偏離已驗證"):
                remote.queue_prompt(graph, pin_models=True)
            remote.call.assert_not_called()
