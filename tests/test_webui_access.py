"""Exercise the local browser boundary without sockets or live credentials."""
import asyncio
import json
import tempfile
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from cstudio import webui as W


HOST = "127.0.0.1:7860"
ORIGIN = f"http://{HOST}"
UI_HEADER = ("x-colab-studio-request", "1")
JSON_HEADER = ("content-type", "application/json")
FILE_URL = "/api/jobs/fixture/out/sample.png"
MULTIPART_HEADER = ("content-type", "multipart/form-data; boundary=fixture")
MULTIPART = (
    b'--fixture\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nSynthetic prompt\r\n'
    b'--fixture\r\nContent-Disposition: form-data; name="seed"\r\n\r\n123\r\n--fixture--\r\n'
)
CONTROLS = (
    ("/api/colab/login/url", (), b""),
    ("/api/colab/login/code", (JSON_HEADER,), b'{"code":"fixture-code","state":"fixture-state"}'),
    ("/api/colab/keep", (JSON_HEADER,), b'{"session":"fixture","minutes":30}'),
    ("/api/colab/cancel", (JSON_HEADER,), b'{"session":"fixture"}'),
    ("/api/colab/stop", (JSON_HEADER,), b'{"session":"fixture"}'),
    ("/api/run/start", (MULTIPART_HEADER,), MULTIPART),
)


class WebUIAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.jobs = root / "jobs"
        self.image = self.jobs / "fixture" / "out" / "sample.png"
        self.image.parent.mkdir(parents=True)
        Image.new("RGBA", (2, 2)).save(self.image)
        self.pending = root / "pending_login.json"
        self.pending.write_text("original pending", encoding="utf-8")
        self.token = root / "token.json"
        self.token.write_text("original token", encoding="utf-8")

        self.cloud = {
            "colab": Mock(side_effect=AssertionError("No live Colab commands")),
            "usage": Mock(return_value=(10.0, "fixture CU")),
            "sessions": Mock(return_value=([], "[colab] No active sessions found on server.")),
            "keep_target": Mock(return_value={}),
            "set_keep": Mock(),
            "start_watchdog": Mock(),
            "request_cancel": Mock(return_value={"state": "confirmed"}),
            "release_runtime": Mock(return_value=W.C.ReleaseState.CONFIRMED),
        }
        self.enterContext(patch.multiple(
            W.C, KEEP=root / "keep", token_ok=Mock(return_value=True),
            account_scope=Mock(return_value="fixture-account"), **self.cloud,
        ))
        self.begin_login = Mock(side_effect=self.fake_begin_login)
        self.finish_login = Mock(side_effect=self.fake_finish_login)
        self.enterContext(patch.multiple(
            W.L, TOKEN=self.token, PENDING=self.pending,
            get_login_url=self.begin_login, submit_login_code=self.finish_login,
        ))
        self.run_manifest = self.enterContext(patch.object(W.R, "run_manifest", return_value={}))
        self.state = W.StudioUIState()
        self.enterContext(patch.multiple(W, JOBS_DIR=self.jobs, STATE=self.state))

    def fake_begin_login(self):
        self.pending.write_text("fixture state", encoding="utf-8")
        return "https://example.test/login", "fixture-state"

    def fake_finish_login(self, code, state):
        self.token.write_text("updated token", encoding="utf-8")
        self.pending.unlink()
        return self.token

    async def request(self, target, *, method="GET", host=HOST, headers=(), body=b"", peer="127.0.0.1", root_path=""):
        path, _, query = target.partition("?")
        fields = ([("host", host)] if host is not None else []) + list(headers)
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "root_path": root_path,
            "query_string": query.encode(),
            "headers": [(key.encode(), value.encode("latin-1")) for key, value in fields],
            "client": (peer, 12345) if peer is not None else None,
            "server": ("127.0.0.1", 7860),
        }
        messages = []
        body_read = False

        async def receive():
            nonlocal body_read
            if body_read:
                await asyncio.Future()
            body_read = True
            return {"type": "http.request", "body": body}

        async def send(message):
            messages.append(message)

        await W.app(scope, receive, send)
        start, *chunks = messages
        return SimpleNamespace(
            status=start["status"], headers=dict(start["headers"]),
            body=b"".join(chunk.get("body", b"") for chunk in chunks), body_read=body_read,
        )

    async def open_ui(self, host=HOST, peer="127.0.0.1"):
        response = await self.request("/", host=host, peer=peer, headers=(("sec-fetch-site", "none"),))
        self.assertEqual(response.status, 200)
        self.assertIn(b"/static/app.js", response.body)
        self.assertEqual(response.headers[b"cache-control"], b"no-store")
        self.assertEqual(response.headers[b"content-security-policy"], b"frame-ancestors 'none'")
        cookies = SimpleCookie(response.headers[b"set-cookie"].decode())
        session, = cookies.values()
        self.assertTrue(session["httponly"])
        self.assertEqual(session["samesite"], "strict")
        return ("cookie", f"{session.key}={session.value}")

    def assert_no_effects(self):
        for fake in (*self.cloud.values(), self.begin_login, self.finish_login, self.run_manifest):
            fake.assert_not_called()
        self.assertEqual(list(self.jobs.iterdir()), [self.image.parents[1]])
        self.assertEqual(self.state.revision, 0)
        self.assertFalse(self.state.running)
        self.assertEqual(self.pending.read_text(encoding="utf-8"), "original pending")
        self.assertEqual(self.token.read_text(encoding="utf-8"), "original token")

    async def test_local_home_assets_and_private_reads(self):
        for host, peer, origin in (
            (HOST, "127.0.0.1", ORIGIN),
            ("LOCALHOST:8765", "127.0.0.1", "http://localhost:8765"),
            ("[::1]:8765", "::1", "http://[::1]:8765"),
            ("localhost:80", "127.0.0.1", "http://localhost"),
        ):
            with self.subTest(host=host):
                cookie = await self.open_ui(host, peer)
                script = await self.request("/static/app.js", host=host, peer=peer)
                self.assertEqual(script.status, 200)
                self.assertNotIn(b"set-cookie", script.headers)
                headers = (cookie, ("sec-fetch-site", "same-origin"))
                for path in ("/api/meta", "/api/run/status"):
                    response = await self.request(path, host=host, peer=peer, headers=headers)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers[b"cache-control"], b"no-store")
                gallery = await self.request("/api/gallery", host=host, peer=peer, headers=headers)
                self.assertEqual(gallery.status, 200)
                self.assertEqual(gallery.headers[b"cache-control"], b"no-store")
                self.assertEqual(json.loads(gallery.body)["items"][0]["artifact_id"], "fixture/out/sample.png")
                image = await self.request(FILE_URL, host=host, peer=peer, headers=headers)
                self.assertEqual(image.status, 200)
                self.assertEqual(image.headers[b"cross-origin-resource-policy"], b"same-origin")
                self.assertEqual(image.headers[b"cache-control"], b"no-store")
                self.assertEqual(image.body, self.image.read_bytes())
                slow = await self.request("/api/colab/status?force=true", host=host, peer=peer, headers=(
                    *headers, ("origin", origin), UI_HEADER,
                ))
                self.assertEqual(slow.status, 200)
                self.assertEqual(json.loads(slow.body)["cu"], 10.0)
        cookie = await self.open_ui()
        browser_poll = await self.request("/api/colab/status", headers=(cookie, UI_HEADER, ("sec-fetch-site", "same-origin")))
        self.assertEqual(browser_poll.status, 200)
        self.cloud["usage"].assert_called_with(account="fixture-account")

    async def test_missing_invalid_and_wrong_port_sessions_cannot_read_or_control(self):
        other_port_cookie = await self.open_ui("127.0.0.1:8765")
        for cookies in ((), (("cookie", "cstudio_ui_7860=invalid"),), (other_port_cookie,)):
            for path in ("/api/meta", "/api/run/status", "/api/colab/status?force=true", "/api/gallery", FILE_URL):
                with self.subTest(path=path, cookies=bool(cookies)):
                    response = await self.request(path, headers=(*cookies, UI_HEADER))
                    self.assertEqual(response.status, 401)
            for path, extra, body in CONTROLS:
                with self.subTest(path=path, cookies=bool(cookies)):
                    response = await self.request(path, method="POST", body=body, headers=(
                        *cookies, ("origin", ORIGIN), UI_HEADER, *extra,
                    ))
                    self.assertEqual(response.status, 401)
                    self.assertFalse(response.body_read)
        self.assert_no_effects()

    async def test_invalid_host_peer_and_proxy_are_rejected_even_for_bootstrap(self):
        cookie = await self.open_ui()
        cases = (
            (400, {"host": None}),
            (400, {"host": "workbench.example.test:7860"}),
            (400, {"host": "127.0.0.1:65536"}),
            (400, {"headers": (("host", HOST),)}),
            (403, {"peer": "192.0.2.10"}),
            (403, {"peer": None}),
            (403, {"headers": (("forwarded", "for=192.0.2.10"),)}),
            (403, {"headers": (("x-forwarded-for", "127.0.0.1"),)}),
            (403, {"headers": (("x-forwarded-host", HOST),)}),
            (403, {"headers": (("x-real-ip", "127.0.0.1"),)}),
        )
        for status, options in cases:
            with self.subTest(options=options):
                home = await self.request("/", **options)
                self.assertEqual(home.status, status)
                self.assertNotIn(b"set-cookie", home.headers)
                headers = (*options.get("headers", ()), cookie, ("origin", ORIGIN), UI_HEADER, MULTIPART_HEADER)
                response = await self.request("/api/run/start", method="POST", body=MULTIPART, **(options | {
                    "headers": headers,
                }))
                self.assertEqual(response.status, status)
                self.assertFalse(response.body_read)
        self.assert_no_effects()

    async def test_session_alone_cannot_authorize_cross_origin_or_unmarked_controls(self):
        cookie = await self.open_ui()
        for headers in (
            (UI_HEADER,),
            (("origin", ORIGIN),),
            (("origin", "null"), UI_HEADER),
            (("origin", "https://example.test"), UI_HEADER),
            (("origin", "http://127.0.0.1:8765"), UI_HEADER),
            (("origin", "https://127.0.0.1:7860"), UI_HEADER),
            (("origin", ORIGIN), ("origin", ORIGIN), UI_HEADER),
            (("origin", ORIGIN), UI_HEADER, UI_HEADER),
        ):
            for path, extra, body in CONTROLS:
                with self.subTest(path=path, headers=headers):
                    response = await self.request(path, method="POST", body=body, headers=(cookie, *headers, *extra))
                    self.assertEqual(response.status, 403)
                    self.assertFalse(response.body_read)
        for path in ("/api/run/status", "/api/colab/status?force=true", "/api/gallery", FILE_URL):
            for headers in (
                (("origin", "https://example.test"), UI_HEADER),
                (("sec-fetch-site", "cross-site"), UI_HEADER),
                (("sec-fetch-site", "same-site"), UI_HEADER),
            ):
                with self.subTest(path=path, headers=headers):
                    response = await self.request(path, headers=(cookie, *headers))
                    self.assertEqual(response.status, 403)
        self.assertEqual((await self.request("/api/colab/status?force=true", headers=(cookie,))).status, 403)
        self.assert_no_effects()

    async def test_asgi_root_path_keeps_private_routes_protected(self):
        cookie = await self.open_ui()
        response = await self.request("/static/api/run/status", root_path="/static")
        self.assertEqual(response.status, 401)
        response = await self.request("/static/api/colab/status", headers=(cookie,), root_path="/static")
        self.assertEqual(response.status, 403)
        self.assert_no_effects()

    async def start_edit(self, cookie, artifact_ids):
        order = json.dumps([{"kind": "artifact", "artifact_id": item} for item in artifact_ids])
        body = (
            '--fixture\r\nContent-Disposition: form-data; name="mode"\r\n\r\nedit\r\n'
            '--fixture\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nSynthetic edit\r\n'
            f'--fixture\r\nContent-Disposition: form-data; name="ref_order"\r\n\r\n{order}\r\n--fixture--\r\n'
        ).encode()
        return await self.request("/api/run/start", method="POST", body=body, headers=(
            cookie, ("origin", ORIGIN), UI_HEADER, MULTIPART_HEADER,
        ))

    async def test_gallery_and_edit_reject_symlink_aliases_without_creating_a_job(self):
        cookie = await self.open_ui()
        outside = self.jobs.parent / "other"
        (outside / "out").mkdir(parents=True)
        Image.new("RGB", (3, 3)).save(outside / "out" / "other.png")
        metadata = outside / "manifest.json"
        metadata.write_text(json.dumps({
            "jobs": [{"id": "sample", "prompt": "private prompt"}],
        }), encoding="utf-8")
        (self.jobs / "other-out").mkdir()
        try:
            (self.jobs / "other-job").symlink_to(outside, target_is_directory=True)
            (self.jobs / "other-out" / "out").symlink_to(outside / "out", target_is_directory=True)
            (self.image.parent / "linked.png").symlink_to(self.image)
            (self.image.parent / "dangling.png").symlink_to(outside / "missing.png")
            (self.image.parents[1] / "manifest.json").symlink_to(metadata)
            (self.image.parent / "results.json").symlink_to(metadata)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links unavailable")
        before = sorted(path.name for path in self.jobs.iterdir())
        response = await self.request("/api/gallery", headers=(cookie,))
        self.assertEqual(response.status, 200)
        items = json.loads(response.body)["items"]
        self.assertEqual([item["artifact_id"] for item in items], ["fixture/out/sample.png"])
        self.assertEqual(items[0]["prompt"], "")
        for artifact in ("other-job/out/other.png", "other-out/out/other.png", "fixture/out/linked.png"):
            with self.subTest(artifact=artifact):
                reply = await self.request(f"/api/jobs/{artifact}", headers=(cookie,))
                self.assertIn(reply.status, (400, 404))
                reply = await self.start_edit(cookie, [artifact])
                self.assertIn(reply.status, (400, 404))
        for artifact in ("/fixture/out/sample.png", "fixture//out/sample.png", r"fixture\out\sample.png"):
            with self.subTest(artifact=artifact):
                self.assertEqual((await self.start_edit(cookie, [artifact])).status, 400)
        reply = await self.request("/api/jobs/fixture/out/results.json", headers=(cookie,))
        self.assertEqual(reply.status, 400)
        self.assertEqual(sorted(path.name for path in self.jobs.iterdir()), before)
        self.run_manifest.assert_not_called()
        self.cloud["colab"].assert_not_called()

    async def test_gallery_edit_copies_exact_artifacts_in_order_and_preserves_alpha(self):
        cookie = await self.open_ui()
        original_job = self.image.parents[1]
        original_ref = original_job / "inputs" / "flat.png"
        original_ref.parent.mkdir()
        Image.new("RGBA", (2, 2), (250, 10, 20, 90)).save(original_ref)
        (original_job / "manifest.json").write_text(json.dumps({"jobs": [{
            "id": "sample", "mode": "edit", "inputs": ["inputs/nested/flat.png", "inputs/flat.png"],
        }]}), encoding="utf-8")
        response = await self.request("/api/gallery", headers=(cookie,))
        item, = json.loads(response.body)["items"]
        restored_ref, = item["inputs"]
        self.assertEqual(restored_ref["artifact_id"], "fixture/inputs/flat.png")
        self.enterContext(patch.object(W, "threading", SimpleNamespace(
            Thread=lambda *, target, **_: SimpleNamespace(start=target),
        )))
        response = await self.start_edit(cookie, [restored_ref["artifact_id"], item["artifact_id"]])
        self.assertEqual(response.status, 202, response.body)
        new_job = self.jobs / json.loads(response.body)["job_dir"]
        manifest = json.loads((new_job / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["jobs"][0]["inputs"], ["inputs/ref_1.png", "inputs/ref_2.png"])
        with Image.open(new_job / "inputs" / "ref_1.png") as first:
            self.assertEqual(first.getpixel((0, 0)), (250, 10, 20, 90))
        with Image.open(new_job / "inputs" / "ref_2.png") as second:
            self.assertEqual(second.getpixel((0, 0)), (0, 0, 0, 0))
        self.run_manifest.assert_called_once()

    async def test_same_origin_login_controls_and_multipart_generation(self):
        cookie = await self.open_ui()
        self.enterContext(patch.object(W, "threading", SimpleNamespace(
            Thread=lambda *, target, **_: SimpleNamespace(start=target),
        )))
        for path, extra, body in CONTROLS:
            with self.subTest(path=path):
                response = await self.request(path, method="POST", body=body, headers=(
                    cookie, ("origin", ORIGIN), ("sec-fetch-site", "same-origin"), UI_HEADER, *extra,
                ))
                self.assertEqual(response.status, 202 if path == "/api/run/start" else 200, response.body)
                self.assertTrue(json.loads(response.body)["ok"])
        job_dir = json.loads(response.body)["job_dir"]
        manifest = self.jobs / job_dir / "manifest.json"
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["jobs"][0]["prompt"], "Synthetic prompt")
        self.run_manifest.assert_called_once()
        self.assertEqual(self.run_manifest.call_args.args[0], manifest)
        self.begin_login.assert_called_once_with()
        self.finish_login.assert_called_once_with("fixture-code", state="fixture-state")
        self.assertFalse(self.pending.exists())
        self.assertEqual(self.token.read_text(encoding="utf-8"), "updated token")
        self.cloud["set_keep"].assert_called_once()
        self.cloud["start_watchdog"].assert_called_once_with("fixture")
        self.cloud["request_cancel"].assert_called_once_with("fixture", None)
        self.cloud["release_runtime"].assert_called_once_with("fixture", owned_only=False, account="fixture-account")

    def test_launcher_binds_only_supported_loopback_hosts_and_disables_forwarding(self):
        with patch("uvicorn.run") as run, patch.object(W.webbrowser, "open") as open_browser, patch("builtins.print"):
            for host in ("0.0.0.0", "::", "192.0.2.10", "workbench.example.test"):
                with self.subTest(host=host), self.assertRaisesRegex(SystemExit, "loopback|只允許"):
                    W.launch_ui(host=host, open_browser=True)
            run.assert_not_called()
            open_browser.assert_not_called()
            for host, expected in (("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"), ("::1", "::1")):
                with self.subTest(host=host):
                    W.launch_ui(host=host, port=8765)
                    run.assert_called_with(W.app, host=expected, port=8765, log_level="info", proxy_headers=False)
