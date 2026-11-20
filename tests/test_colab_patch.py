"""Verify the pinned Colab CLI patches using package copies and a loopback HTTP peer."""
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.machinery import PathFinder
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

from cstudio import patch_colab_cli as P


class ColabPatchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        dist = importlib.metadata.distribution(P.DIST)
        installed_library = Path(dist.locate_file("")).resolve()
        self.pristine = {
            name: Path(dist.locate_file(f"colab_cli/{name}")).read_text(encoding="utf-8")
            for name in P.SOURCE_HASHES
        }
        self.pristine["client.py"] = (
            self.pristine["client.py"].replace(P.CLIENT_FIX + P.CLIENT_ANCHOR, P.CLIENT_ANCHOR)
            .replace(P.LEGACY_CLIENT_FIX + P.CLIENT_ANCHOR, P.CLIENT_ANCHOR)
        )
        self.pristine["console.py"] = (
            self.pristine["console.py"].replace(P.CONSOLE_IMPORT_FIX, P.CONSOLE_IMPORTS)
            .replace(P.CONSOLE_TTY_FIX, P.CONSOLE_TTY)
        )
        self.prefix = self.root / "active-env"
        self.library = self.prefix / "site-packages"
        self.package = self.library / "colab_cli"
        self.package.mkdir(parents=True)
        for name, source in self.pristine.items():
            (self.package / name).write_text(source, encoding="utf-8")
        self.metadata = self.library / "google_colab_cli-0.7.4.dist-info" / "METADATA"
        self.metadata.parent.mkdir()
        self.set_version(P.VERSION)
        self.home = self.root / "fixture-home"
        self.settings = self.home / ".config" / "colab-cli" / "settings.json"
        self.enterContext(patch.object(Path, "home", return_value=self.home))
        self.enterContext(patch.object(sys, "prefix", str(self.prefix)))
        # Leave dependencies available but pass only fixture namespaces to the package lookup.
        self.enterContext(patch.object(sys, "path", [str(self.library), *sys.path]))
        self.enterContext(patch.object(P.sysconfig, "get_path", return_value=str(self.library)))
        self.enterContext(patch.object(P.importlib.util, "find_spec", side_effect=lambda name:
            PathFinder.find_spec(name, [p for p in sys.path if Path(p).resolve() != installed_library])))

    def set_version(self, version):
        self.metadata.write_text(f"Name: google-colab-cli\nVersion: {version}\n", encoding="utf-8")

    def run_patch(self, *args):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as errors:
            try:
                P.main(args)
            except SystemExit as exc:
                return exc.code, out.getvalue(), errors.getvalue()
        return 0, out.getvalue(), errors.getvalue()

    def load_copy(self, filename):
        name = "fixture_colab_" + Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, self.package / filename)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)
        return module

    def test_status_apply_and_repeat_leave_settings_and_other_environments_alone(self):
        self.settings.parent.mkdir(parents=True)
        settings = b'{"enable_update_check": true, "custom": "keep"}\n'
        self.settings.write_bytes(settings)
        linked = self.root / "other-env" / "colab_cli" / "client.py"
        linked.parent.mkdir(parents=True)
        client = self.package / "client.py"
        if os.name == "posix":
            client.chmod(0o640)
        os.link(client, linked)

        code, out, _ = self.run_patch("--status")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out), {
            "version": P.VERSION, "package": str(self.package),
            "files": {"client.py": "unpatched", "console.py": "unpatched"},
        })
        self.assertTrue(client.samefile(linked))
        self.assertEqual(self.run_patch()[0], 0)
        self.assertFalse(client.samefile(linked))
        self.assertEqual(linked.read_text(encoding="utf-8"), self.pristine["client.py"])
        self.assertEqual(self.settings.read_bytes(), settings)
        if os.name == "posix":
            self.assertEqual(client.stat().st_mode & 0o777, 0o640)
        after_patch = {name: (self.package / name).stat() for name in self.pristine}
        self.assertEqual(self.run_patch("--status")[0], 0)
        self.assertEqual(self.run_patch()[0], 0)
        for name, before in after_patch.items():
            after = (self.package / name).stat()
            self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))

    def test_recognizes_both_client_markers_and_upgrades_legacy_console(self):
        for client_fix in (P.CLIENT_FIX, P.LEGACY_CLIENT_FIX):
            with self.subTest(client_fix=client_fix):
                client = self.package / "client.py"
                # Old Windows installations can use CRLF line endings.
                source = self.pristine["client.py"].replace(P.CLIENT_ANCHOR, client_fix + P.CLIENT_ANCHOR)
                old_client = source.replace("\n", "\r\n").encode()
                client.write_bytes(old_client)
                self.package.joinpath("console.py").write_text(
                    self.pristine["console.py"].replace(P.CONSOLE_IMPORTS, P.CONSOLE_IMPORT_FIX),
                    encoding="utf-8",
                )
                status, code = P.patch_status()
                self.assertEqual(code, 1)
                self.assertEqual(status["files"], {"client.py": "patched", "console.py": "legacy"})
                self.assertEqual(self.run_patch()[0], 0)
                self.assertEqual(P.patch_status()[1], 0)
                self.assertEqual(client.read_bytes(), old_client)

    def test_mismatched_version_and_import_shadowing_leave_files_alone(self):
        self.set_version("0.7.5")
        self.assertEqual(self.run_patch("--status")[0], 2)
        self.assertIn("0.7.5", self.run_patch()[2])
        self.set_version(P.VERSION)
        shadow = self.root / "shadow" / "colab_cli"
        shadow.mkdir(parents=True)
        with patch.object(sys, "path", [str(shadow.parent), *sys.path]):
            self.assertEqual(self.run_patch("--status")[0], 2)
            self.assertEqual(self.run_patch()[0], 2)
            (shadow / "__init__.py").write_text("raise AssertionError('inspection imported a shadow package')\n")
            self.assertEqual(self.run_patch("--status")[0], 2)
        for name, source in self.pristine.items():
            self.assertEqual((self.package / name).read_text(encoding="utf-8"), source)
        self.assertFalse(self.settings.exists())

    def test_changed_anchor_or_marker_alone_fails_before_either_file_is_replaced(self):
        for name, damaged in (
            ("client.py", self.pristine["client.py"].replace(P.CLIENT_ANCHOR, "        response = other_request(")),
            ("client.py", self.pristine["client.py"] + "\n# h3skill: 411 fix\n"),
            ("console.py", self.pristine["console.py"].replace("import termios", "termios = None")),
            ("console.py", self.pristine["console.py"].replace(P.CONSOLE_TTY, P.CONSOLE_TTY + P.CONSOLE_TTY)),
        ):
            with self.subTest(name=name):
                path = self.package / name
                path.write_text(damaged, encoding="utf-8")
                status, code = P.patch_status()
                self.assertEqual(code, 2)
                self.assertEqual(status["files"][name], "incompatible")
                self.assertEqual(self.run_patch()[0], 2)
                for filename, source in self.pristine.items():
                    self.assertEqual((self.package / filename).read_text(encoding="utf-8"),
                                     damaged if filename == name else source)
                path.write_text(self.pristine[name], encoding="utf-8")
        self.assertFalse(self.settings.exists())

    def test_partial_publish_keeps_complete_files_and_can_be_retried(self):
        replace = Path.replace

        def refuse_console(path, target):
            if target.name == "console.py":
                raise PermissionError("fixture replacement denied")
            return replace(path, target)

        with patch.object(Path, "replace", refuse_console):
            code, _, errors = self.run_patch()
        self.assertEqual(code, 2)
        self.assertIn("fixture replacement denied", errors)
        self.assertEqual(P.patch_status()[0]["files"], {"client.py": "patched", "console.py": "unpatched"})
        self.assertEqual(list(self.package.glob(".*-*")), [])
        self.assertEqual(self.run_patch()[0], 0)
        self.assertEqual(P.patch_status()[1], 0)
        self.assertFalse(self.settings.exists())

    def test_console_import_and_piped_use_with_missing_posix_modules(self):
        with patch.dict(sys.modules, {"termios": None, "tty": None}):
            with self.assertRaises(ImportError):
                self.load_copy("console.py")
            self.assertEqual(self.run_patch()[0], 0)
            console = self.load_copy("console.py")
        socket = Mock()
        session = Mock(url="https://example.invalid", token="fixture")
        stdin = Mock()
        with patch.object(console.websocket, "WebSocketApp", return_value=socket) as connect, \
                patch.object(console.sys, "stdin", stdin), redirect_stdout(io.StringIO()):
            stdin.isatty.return_value = False
            console.connect_console(session)
            socket.run_forever.assert_called_once()
            stdin.isatty.return_value = True
            with self.assertRaisesRegex(RuntimeError, "requires a POSIX terminal"):
                console.connect_console(session)
            stdin.fileno.assert_not_called()
            connect.assert_called_once()

    def test_http_411_workaround_preserves_get_and_explicit_post_bodies(self):
        received = []

        class Peer(BaseHTTPRequestHandler):
            def exchange(self):
                length = self.headers.get("Content-Length")
                body = self.rfile.read(int(length or "0"))
                received.append((self.command, self.path, length, self.headers.get("Content-Type"), body))
                self.send_response(411 if self.command == "POST" and not body else 204)
                self.end_headers()

            do_GET = do_POST = exchange

            def log_message(self, *_args):
                pass

        with ExitStack() as cleanup:
            server = cleanup.enter_context(HTTPServer(("127.0.0.1", 0), Peer))
            transport = cleanup.enter_context(AuthorizedSession(Credentials("fixture")))
            transport.trust_env = False
            worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            cleanup.callback(worker.join)
            cleanup.callback(server.shutdown)
            url = f"http://127.0.0.1:{server.server_port}"
            pristine = self.load_copy("client.py")
            before = pristine.Client(pristine.ColabEnvironment(url, url), transport)
            with self.assertRaises(pristine.ColabRequestError) as err:
                before._issue_request(url + "/assign", method="POST", timeout=3)
            self.assertEqual(err.exception.response.status_code, 411)
            self.assertEqual(self.run_patch()[0], 0)
            patched = self.load_copy("client.py")
            after = patched.Client(patched.ColabEnvironment(url, url), transport)
            after._issue_request(url + "/assign", method="pOsT", timeout=3)
            for client in (before, after):
                client._issue_request(url + "/list", timeout=3)
                client._issue_request(url + "/json", method="POST", json={"fixture": 1}, timeout=3)
                client._issue_request(url + "/raw", method="POST", data=b"fixture raw", timeout=3)
                client._issue_request(url + "/form", method="POST", data={"fixture": "form"}, timeout=3)
                client._issue_request(url + "/file", method="POST", files={"fixture": ("a.txt", b"file")}, timeout=3)

        self.assertEqual(received[:2], [
            ("POST", "/assign?authuser=0", "0", None, b""),
            ("POST", "/assign?authuser=0", "2", "application/json", b"{}"),
        ])
        self.assertEqual(received[2:6], received[7:11])
        for item in (received[6], received[11]):
            method, path, _, content_type, body = item
            self.assertEqual((method, path), ("POST", "/file?authuser=0"))
            self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
            self.assertIn(b'name="fixture"; filename="a.txt"\r\n\r\nfile\r\n', body)
