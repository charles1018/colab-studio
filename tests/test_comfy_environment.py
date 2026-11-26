"""Offline checks for ComfyUI installation and service reuse in a warm runtime."""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from remote import remote_runner as remote


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.comfy = self.root / "ComfyUI"
        self.node = self.comfy / "custom_nodes" / "ComfyUI-GGUF"
        self.marker = self.root / ".cstudio_setup_v1"
        self.comfy_origin = self.make_repo("comfy-origin", "torch\nfixture-lib>=1\n")
        self.node_origin = self.make_repo("gguf-origin", "gguf>=0.13\n")
        self.pins = {
            "comfyui": {"repo": str(self.comfy_origin), "commit": self.head(self.comfy_origin)},
            "custom_nodes": [{"name": self.node.name, "repo": str(self.node_origin), "commit": self.head(self.node_origin)}],
            "pip": [{"name": "fixture-lib", "spec": "fixture-lib>=1", "version": None}],
        }
        torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda _: "fixture", get_device_capability=lambda _: (8, 9),
            get_device_properties=lambda _: SimpleNamespace(total_memory=24 * 1024**3),
        ))
        self.versions = {"torch": "2.7.1", "fixture-lib": "1.5", "gguf": "0.13.1"}
        self.stop = Mock()
        self.pip = Mock(side_effect=AssertionError("real pip is disabled"))
        self.setup = remote.setup_command
        self.enterContext(patch.multiple(
            remote, COMFY=self.comfy, JOB=self.root, MARK=self.marker, STATE={},
            COMFY_REPO=str(self.comfy_origin), GGUF_REPO=str(self.node_origin), EXTRA_PIP=("fixture-lib>=1",),
            log=Mock(), save=Mock(), _stop_comfy_for_install=self.stop,
            setup_command=self.setup_command,
            version=self.read_version,
        ))
        self.enterContext(patch.dict(sys.modules, {"torch": torch}))

    def setup_command(self, args, **opts):
        if args[0] == "git":
            opts.setdefault("capture_output", True)
            return self.setup(args, **opts)
        return self.pip(args, **opts)

    def read_version(self, package):
        try:
            return self.versions[package]
        except KeyError as exc:
            raise remote.PackageNotFoundError(package) from exc

    def git(self, path, *args):
        return subprocess.check_output(
            ["git", "-C", str(path), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
             "-c", "commit.gpgsign=false", *args],
            text=True, stderr=subprocess.PIPE,
        ).strip()

    def head(self, path):
        return self.git(path, "rev-parse", "HEAD")

    def make_repo(self, name, requirements):
        repo = self.root / name
        repo.mkdir()
        self.git(repo, "init", "-q")
        (repo / ".gitignore").write_text("/custom_nodes/\n", encoding="utf-8")
        (repo / "requirements.txt").write_text(requirements, encoding="utf-8")
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-qm", "fixture")
        return repo

    def commit_upstream_node(self, content):
        rel = Path("custom_nodes/websocket_image_save.py")
        node = self.comfy_origin / rel
        node.parent.mkdir(exist_ok=True)
        node.write_text(content, encoding="utf-8")
        self.git(self.comfy_origin, "add", "-f", str(rel))
        self.git(self.comfy_origin, "commit", "-qm", "upstream node fixture")
        self.pins["comfyui"]["commit"] = self.head(self.comfy_origin)
        return self.comfy / rel

    def test_cold_then_warm_setup_verifies_actual_checkouts_and_versions(self):
        node = self.commit_upstream_node("# Stock ComfyUI node\n")
        signature = remote.install(self.pins)
        marker = json.loads(self.marker.read_text(encoding="utf-8"))
        self.assertEqual(marker["schema"], 2)
        self.assertEqual(marker["signature"], signature)
        self.assertEqual(marker["environment"]["packages"], self.versions)
        self.assertEqual(self.head(self.node), self.pins["custom_nodes"][0]["commit"])
        self.assertEqual(self.head(self.comfy), self.pins["comfyui"]["commit"])
        self.assertEqual(node.read_text(encoding="utf-8"), "# Stock ComfyUI node\n")
        self.assertEqual(remote.install(self.pins), signature)
        self.stop.assert_called_once()
        self.pip.assert_not_called()

    def test_pinned_update_revalidates_the_stock_node_at_the_new_revision(self):
        node = self.commit_upstream_node("# Original upstream node\n")
        first = remote.install(self.pins)
        self.stop.reset_mock()

        self.commit_upstream_node("# Updated upstream node\n")
        updated = remote.install(self.pins)
        self.assertNotEqual(updated, first)
        self.assertEqual(node.read_text(encoding="utf-8"), "# Updated upstream node\n")
        self.assertEqual(remote.install(self.pins), updated)
        self.stop.assert_called_once()
        self.pip.assert_not_called()

    def test_changed_stock_node_is_rejected_even_when_git_status_is_clean(self):
        node = self.commit_upstream_node("# Stock ComfyUI node\n")
        remote.install(self.pins)
        self.stop.reset_mock()
        self.git(self.comfy, "update-index", "--assume-unchanged", node.relative_to(self.comfy).as_posix())
        node.write_text("# Local change\n", encoding="utf-8")
        self.assertEqual(self.git(self.comfy, "status", "--porcelain"), "")

        with self.assertRaisesRegex(RuntimeError, "未列入.*websocket_image_save.py"):
            remote.install(self.pins)
        self.assertEqual(node.read_text(encoding="utf-8"), "# Local change\n")
        self.stop.assert_not_called()
        self.pip.assert_not_called()

    def test_legacy_marker_and_reenabled_pins_use_real_head(self):
        first = remote.install(self.pins)
        before = self.pins["comfyui"]["commit"]
        self.git(self.comfy_origin, "commit", "--allow-empty", "-qm", "second revision")
        after = self.head(self.comfy_origin)
        self.marker.write_text(json.dumps({"comfyui": before}), encoding="utf-8")
        self.pins["comfyui"]["commit"] = after
        self.assertNotEqual(remote.install(self.pins), first)
        self.assertEqual(self.head(self.comfy), after)

        self.stop.reset_mock()
        remote.install(None)
        self.assertEqual(self.head(self.comfy), after)
        self.stop.assert_not_called()
        self.pins["comfyui"]["commit"] = before
        self.assertEqual(remote.install(self.pins), first)
        self.stop.assert_called_once()

    def test_dependency_changes_require_a_new_signature_or_install(self):
        first = remote.install(self.pins)
        self.stop.reset_mock()
        self.versions["fixture-lib"] = "1.8"
        changed = remote.install(self.pins)
        self.assertNotEqual(first, changed)
        self.stop.assert_not_called()
        self.pip.assert_not_called()

        self.pins["pip"][0]["version"] = "1.5"
        self.pip.side_effect = lambda *_args, **_kw: self.versions.update({"fixture-lib": "1.5"})
        self.assertEqual(remote.install(self.pins), first)
        args = self.pip.call_args.args[0]
        self.assertIn("fixture-lib==1.5", args)
        self.assertIn(str(self.node / "requirements.txt"), args)
        self.stop.assert_called_once()

    def test_missing_dependency_and_cancelled_install_preserve_marker_and_retry(self):
        signature = remote.install(self.pins)
        good_record = self.marker.read_bytes()
        del self.versions["gguf"]
        self.pip.side_effect = remote.BatchCancelled
        with self.assertRaises(remote.BatchCancelled):
            remote.install(self.pins)
        self.assertEqual(self.marker.read_bytes(), good_record)
        self.pip.side_effect = lambda *_args, **_kw: self.versions.update({"gguf": "0.13.1"})
        self.assertEqual(remote.install(self.pins), signature)

    def test_failed_fetch_or_checkout_keeps_previous_marker_and_head(self):
        signature = remote.install(self.pins)
        good_record = self.marker.read_bytes()
        previous = self.head(self.comfy)
        self.pins["comfyui"]["commit"] = "a" * 40
        with self.assertRaises(subprocess.CalledProcessError):
            remote.install(self.pins)
        self.assertEqual(self.marker.read_bytes(), good_record)
        self.assertEqual(self.head(self.comfy), previous)
        self.pins["comfyui"]["commit"] = previous
        self.assertEqual(remote.install(self.pins), signature)

    def test_dirty_checkout_foreign_origin_and_extra_code_are_preserved(self):
        remote.install(self.pins)
        self.stop.reset_mock()
        req = self.comfy / "requirements.txt"
        req.write_text("locally edited", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "未提交"):
            remote.install(self.pins)
        self.assertEqual(req.read_text(encoding="utf-8"), "locally edited")
        self.git(self.comfy, "checkout", "--", "requirements.txt")
        self.git(self.node, "remote", "set-url", "origin", str(self.comfy_origin))
        with self.assertRaisesRegex(RuntimeError, "Git origin"):
            remote.install(self.pins)
        self.git(self.node, "remote", "set-url", "origin", str(self.node_origin))
        extra_file = self.node.parent / "unlisted.py"
        extra_file.touch()
        with self.assertRaisesRegex(RuntimeError, "未列入.*unlisted.py"):
            remote.install(self.pins)
        extra_file.rename(extra_file.with_suffix(".disabled"))
        extra = self.node.parent / "extra"
        extra.mkdir()
        (extra / "prestartup_script.py").touch()
        with self.assertRaisesRegex(RuntimeError, "未列入"):
            remote.install(self.pins)
        self.assertTrue((extra / "prestartup_script.py").is_file())
        extra.rename(extra.with_suffix(".disabled"))
        remote.install(self.pins)
        self.stop.assert_not_called()
        self.pip.assert_not_called()

    def test_requested_versions_require_complete_file_even_when_models_opt_out(self):
        with self.assertRaisesRegex(RuntimeError, "versions.json"):
            remote.load_pins({"_pin_models": False})
        self.assertIsNone(remote.load_pins({"_pin_versions": False}))
        with self.assertRaisesRegex(RuntimeError, "不完整"):
            remote.install({})
        self.stop.assert_not_called()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.comfy = root / "ComfyUI"
        self.marker = root / "comfy-process.json"
        self.log = root / "comfyui.log"
        self.record = {"pid": 500, "token": "1" * 32, "environment": "installed"}
        self.args = [str(self.comfy / "main.py"), "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"]
        self.probe = Mock()
        self.owns = Mock(side_effect=lambda record: record == self.record)
        self.listener = Mock(return_value=True)
        self.call = Mock(return_value={"queue_running": [], "queue_pending": []})
        self.enterContext(patch.multiple(
            remote, COMFY=self.comfy, COMFY_PROCESS=self.marker, COMFY_LOG=self.log,
            STATE={}, COMFY_URL=remote.DEFAULT_COMFY_URL,
            log=Mock(), call=self.call, _comfy_system=self.probe_system, _owns_comfy_process=self.owns,
            _comfy_listener_sockets=Mock(return_value={"fixture-socket"}), _process_has_listener=self.listener,
        ))
        self.spawn = self.enterContext(patch.object(remote.subprocess, "Popen"))

    def probe_system(self):
        args = self.probe()
        return None if args is None else {"args": args, "versions": {}}

    def test_batch_launch_uses_weights_and_canvas_to_choose_vram_mode(self):
        root = self.comfy.parent
        inputs = root / "inputs"
        inputs.mkdir()
        for name, size in (("ref.png", (768, 512)), ("large.png", (2048, 2048))):
            Image.new("RGB", size).save(inputs / name)
        versions = Path(remote.__file__).with_name("versions.json").read_text(encoding="utf-8")
        (root / "versions.json").write_text(versions, encoding="utf-8")
        pins = json.loads(versions)
        sizes = {(model["repo"], model["file"]): model["bytes"] for model in pins["models"]}
        t2i = {"id": "t2i", "prompt": "test"}
        edit = {"id": "edit", "prompt": "edit", "mode": "edit", "inputs": ["ref.png", "ref.png"]}
        bf16 = dict(t2i, id="bf16", quant="BF16")
        mixed = [t2i, bf16]
        cases = (
            ("L4 1K mixed quantizations", 22, {}, [dict(t2i, id=q.lower(), quant=q) for q in ("Q2_K", "Q4_K_M", "INT8")], True),
            ("L4 1K Edit", 22, {"quant": "Q4_K_M"}, [edit], True),
            ("L4 BF16 alone", 22, {}, [bf16], False),
            ("L4 BF16 in mixed batch", 22, {}, mixed, False),
            ("L4 CLI quant override", 22, {"_cli_quant": "INT8"}, mixed, True),
            ("L4 INT8 2K", 22, {"resolution": "2K"}, [t2i], False),
            ("L4 Edit first image at original size", 22, {}, [dict(edit, edit_resolution=0)], True),
            ("L4 Edit large original size", 22, {}, [dict(edit, inputs=["large.png"], edit_resolution=0)], False),
            ("L4 Edit large custom canvas", 22, {}, [dict(edit, width=2048, height=2048)], False),
            ("24 GiB INT8 2K with W4A8 encoder", 24, {"text_encoder_variant": "w4a8"}, [dict(t2i, resolution="2K")], True),
            ("24 GiB INT8 2K with default encoder", 24, {}, [dict(t2i, resolution="2K")], False),
            ("40 GiB BF16", 40, {}, [bf16], True),
            ("40 GiB BF16 with PE", 40, {"refine_prompt": True}, mixed, False),
        )
        for label, vram, settings, jobs, gpu_only in cases:
            with self.subTest(label=label):
                (root / "manifest.json").write_text(json.dumps({**settings, "jobs": jobs}), encoding="utf-8")
                self.spawn.reset_mock()
                self.spawn.side_effect = RuntimeError("stubbed ComfyUI startup")
                self.probe.return_value = None
                with patch.multiple(
                    remote, JOB=root, MANAGED_RUN_ID=None, STATE={"gpu_info": {"vram_gib": vram}},
                    save=Mock(), install=Mock(return_value="installed"),
                    download_models=Mock(side_effect=lambda files, **_: {
                        model: {"bytes": sizes[model[:2]]} for model in files
                    }),
                ), self.assertRaisesRegex(RuntimeError, "stubbed ComfyUI startup"):
                    remote.run_batch()
                flags = ["--gpu-only"] if gpu_only else []
                self.spawn.assert_called_once()
                self.assertEqual(self.spawn.call_args.args[0], [sys.executable, *self.args, *flags])

    def test_matching_owned_service_is_reused(self):
        remote._write_record(self.marker, self.record)
        self.probe.return_value = [*self.args, "--gpu-only"]
        remote.start_comfy(gpu_only=True, environment="installed")
        self.spawn.assert_not_called()
        self.call.assert_not_called()

    def test_owned_service_restarts_for_vram_mode_and_environment_changes(self):
        for running, gpu_only, environment, flags in (
            ([*self.args, "--gpu-only"], False, "installed", []),
            (self.args, True, "installed", ["--gpu-only"]),
            ([*self.args, "--gpu-only"], True, "changed", ["--gpu-only"]),
        ):
            with self.subTest(gpu_only=gpu_only, environment=environment), patch.object(remote, "_stop_owned_comfy") as stop:
                remote._write_record(self.marker, self.record)
                self.spawn.reset_mock()
                self.spawn.return_value.pid = 501
                self.spawn.return_value.poll.return_value = None
                self.probe.side_effect = [running, [*self.args, *flags]]
                remote.start_comfy(gpu_only=gpu_only, environment=environment)
                stop.assert_called_once_with(self.record)
                self.assertEqual(self.spawn.call_args.args[0], [sys.executable, *self.args, *flags])
                launch = json.loads(self.marker.read_text(encoding="utf-8"))
                self.assertEqual(launch["environment"], environment)
                self.assertEqual(self.spawn.call_args.kwargs["env"][remote.COMFY_INSTANCE], launch["token"])

    def test_busy_or_unknown_queue_prevents_restart(self):
        remote._write_record(self.marker, self.record)
        self.probe.return_value = [*self.args, "--gpu-only"]
        with patch.object(remote.os, "pidfd_open", create=True) as pidfd:
            for queue in ({"queue_running": [[1]], "queue_pending": []}, {"queue_running": []}):
                with self.subTest(queue=queue), self.assertRaisesRegex(RuntimeError, "佇列"):
                    self.call.return_value = queue
                    remote.start_comfy(gpu_only=False, environment="installed")
        pidfd.assert_not_called()
        self.spawn.assert_not_called()

    def test_external_service_cannot_be_stopped_or_used_with_mismatched_flags(self):
        self.probe.return_value = ["main.py", *self.args[1:], "--gpu-only"]
        with patch.object(remote, "_stop_owned_comfy") as stop:
            with self.assertRaisesRegex(RuntimeError, "無法確認程序"):
                remote._stop_comfy_for_install()
            with self.assertRaisesRegex(RuntimeError, "啟動參數"):
                remote.start_comfy(gpu_only=False, environment="installed")
            with patch.object(remote, "_legacy_comfy_checkout", return_value=True):
                remote.start_comfy(gpu_only=True, environment="installed")
            stop.assert_not_called()
        self.spawn.assert_not_called()

    def test_process_id_reused_after_probe_does_not_receive_signal(self):
        remote._write_record(self.marker, self.record)
        self.probe.return_value = [*self.args, "--gpu-only"]
        self.owns.side_effect = [True, False]
        with patch.object(remote.os, "pidfd_open", return_value=17, create=True), \
             patch.object(remote.os, "close") as close, \
             patch.object(remote.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaisesRegex(RuntimeError, "身分已變更"):
                remote.start_comfy(gpu_only=False, environment="installed")
        send.assert_not_called()
        close.assert_called_once_with(17)
        self.spawn.assert_not_called()

    def test_stops_only_the_verified_process_and_confirms_exit(self):
        self.probe.return_value = None
        with patch.object(remote.os, "pidfd_open", return_value=17, create=True) as open_pidfd, \
             patch.object(remote.os, "close") as close, \
             patch.object(remote.signal, "pidfd_send_signal", create=True) as send, \
             patch.object(remote.select, "select", return_value=([17], [], [])):
            remote._stop_owned_comfy(self.record)
        open_pidfd.assert_called_once_with(500)
        send.assert_called_once_with(17, signal.SIGTERM)
        close.assert_called_once_with(17)
        self.probe.assert_called_once()

    def test_wrong_listener_and_unknown_health_block_changes(self):
        remote._write_record(self.marker, self.record)
        self.probe.return_value = self.args
        self.listener.return_value = False
        with self.assertRaisesRegex(RuntimeError, "由 Studio 啟動"):
            remote.start_comfy(gpu_only=False, environment="installed")
        self.probe.side_effect = TimeoutError("fixture timeout")
        with self.assertRaises(TimeoutError):
            remote._stop_comfy_for_install()
        self.call.assert_not_called()
        self.spawn.assert_not_called()


class ServiceProbeTests(unittest.TestCase):
    def test_only_explicit_refusal_counts_as_absent(self):
        with patch.object(remote, "call", side_effect=urllib.error.URLError(ConnectionRefusedError())):
            self.assertIsNone(remote._comfy_system())
        with patch.object(remote, "call", side_effect=urllib.error.URLError(TimeoutError())):
            with self.assertRaises(urllib.error.URLError):
                remote._comfy_system()
        with patch.object(remote, "call", return_value={"system": {}}):
            with self.assertRaisesRegex(RuntimeError, "啟動參數"):
                remote._comfy_system()

    @unittest.skipUnless(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "requires Linux pidfds")
    def test_local_http_process_is_reused_then_restarted_with_verified_socket(self):
        try:
            os.close(os.pidfd_open(os.getpid()))
        except OSError as exc:
            self.skipTest(f"pidfd unavailable: {exc}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / "main.py").write_text(
            "import json, sys\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        system = {'argv': sys.argv, 'comfyui_version': 'fixture', 'python_version': sys.version,\n"
            "                  'pytorch_version': 'fixture+cu', 'ram_free': 42}\n"
            "        data = {'system': system} if self.path == '/system_stats' else {\n"
            "            'queue_running': [], 'queue_pending': []}\n"
            "        self.send_response(200)\n"
            "        self.end_headers()\n"
            "        self.wfile.write(json.dumps(data).encode())\n"
            "HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), Handler).serve_forever()\n",
            encoding="utf-8",
        )
        spawn = subprocess.Popen
        children = []

        def launch(*args, **kwargs):
            child = spawn(*args, **kwargs)
            children.append(child)
            return child

        def cleanup():
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

        self.addCleanup(cleanup)
        clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: time.sleep(0.01))
        opener = remote.urllib.request.build_opener(remote.urllib.request.ProxyHandler({}))
        with patch.multiple(
            remote, COMFY=root, COMFY_PROCESS=root / "process.json", COMFY_LOG=root / "comfy.log",
            COMFY_URL=f"http://127.0.0.1:{port}", time=clock, log=Mock(), STATE={},
        ), patch.object(subprocess, "Popen", side_effect=launch), \
             patch.object(remote.urllib.request, "urlopen", side_effect=lambda req, **kw: opener.open(req, **kw)):
            running = remote.start_comfy(gpu_only=True, environment="fixture")
            self.assertEqual(running["versions"], {
                "comfyui_version": "fixture", "python_version": sys.version, "pytorch_version": "fixture+cu",
            })
            self.assertTrue(running["verified"])
            remote.start_comfy(gpu_only=True, environment="fixture")
            self.assertEqual(len(children), 1)
            self.assertIn("--gpu-only", running["args"])
            remote.start_comfy(gpu_only=False, environment="fixture")
            self.assertEqual(len(children), 2)
            self.assertEqual(children[0].wait(timeout=5), -signal.SIGTERM)
            self.assertNotIn("--gpu-only", remote._comfy_system()["args"])
