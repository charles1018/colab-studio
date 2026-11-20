"""Exercise the local browser boundary without sockets or live credentials."""
import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlencode

from PIL import Image

from cstudio import web_inputs as U
from cstudio import webui as W
from remote.manifest import MAX_SEED


HOST = "127.0.0.1:7860"
ORIGIN = f"http://{HOST}"
UI_HEADER = ("x-colab-studio-request", "1")
JSON_HEADER = ("content-type", "application/json")
FILE_URL = "/api/jobs/fixture/out/sample.png"
THUMBNAIL_URL = "/api/gallery/thumbnail/fixture/sample.png"
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


def image_bytes(mode="RGB", size=(3, 4), color=None, format="PNG"):
    with io.BytesIO() as output, Image.new(mode, size, color) as image:
        image.save(output, format=format)
        return output.getvalue()


def edit_form(files=(), *, fields=()):
    parts = [("mode", "edit"), ("prompt", "Synthetic edit"), *fields]
    body = b"".join(
        f'--fixture\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        for key, value in parts
    )
    for index, file in enumerate(files):
        body += (
            f'--fixture\r\nContent-Disposition: form-data; name="ref_files"; filename="ref{index}.png"\r\n\r\n'.encode()
            + file + b"\r\n"
        )
    return body + b"--fixture--\r\n"


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
        self.worker_threads = []
        self.wait_for_workers = False
        self.state = W.StudioUIState()
        self.enterContext(patch.multiple(W, JOBS_DIR=self.jobs, STATE=self.state))

    def use_worker_threads(self, *, wait=True):
        def tracked_thread(**kwargs):
            thread = threading.Thread(**kwargs)
            self.worker_threads.append(thread)
            return thread

        self.wait_for_workers = wait
        self.enterContext(patch.object(W, "threading", SimpleNamespace(Thread=tracked_thread)))
        self.addCleanup(self.join_workers)

    def join_workers(self):
        for thread in self.worker_threads:
            thread.join(5)
            self.assertFalse(thread.is_alive(), "fixture worker did not finish")

    def assert_clean_staging(self):
        self.assertEqual(list((self.jobs / U.STAGING_DIR).glob("pending-*")), [])

    def fake_begin_login(self):
        self.pending.write_text("fixture state", encoding="utf-8")
        return "https://example.test/login", "fixture-state"

    def fake_finish_login(self, code, state):
        self.token.write_text("updated token", encoding="utf-8")
        self.pending.unlink()
        return self.token

    async def request(
        self, target, *, method="GET", host=HOST, headers=(), body=b"", peer="127.0.0.1", root_path="",
        before_read=None,
    ):
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
        incoming = (body,) if isinstance(body, bytes) else body
        reads = 0

        async def receive():
            nonlocal reads
            if before_read:
                await before_read(reads)
            if reads == len(incoming):
                await asyncio.Future()
            data = incoming[reads]
            reads += 1
            if data is None:
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": data, "more_body": reads < len(incoming)}

        async def send(message):
            messages.append(message)

        await W.app(scope, receive, send)
        if self.wait_for_workers:
            await asyncio.to_thread(self.join_workers)
        start, *chunks = messages
        return SimpleNamespace(
            status=start["status"], headers=dict(start["headers"]),
            body=b"".join(chunk.get("body", b"") for chunk in chunks), body_read=bool(reads), reads=reads,
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
        self.assertEqual([path for path in self.jobs.iterdir() if path.name != U.STAGING_DIR],
                         [self.image.parents[1]])
        self.assert_clean_staging()
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
            for path in (
                "/api/meta", "/api/run/status", "/api/colab/status?force=true", "/api/gallery", FILE_URL, THUMBNAIL_URL,
            ):
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
        for path in ("/api/run/status", "/api/colab/status?force=true", "/api/gallery", FILE_URL, THUMBNAIL_URL):
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

    async def submit_form(self, cookie, body, *, headers=(), before_read=None):
        return await self.request("/api/run/start", method="POST", body=body, before_read=before_read, headers=(
            cookie, ("origin", ORIGIN), UI_HEADER, MULTIPART_HEADER, *headers,
        ))

    async def start_edit(self, cookie, artifact_ids):
        order = json.dumps([{"kind": "artifact", "artifact_id": item} for item in artifact_ids])
        return await self.submit_form(cookie, edit_form(fields=(("ref_order", order),)))

    async def test_upload_content_length_over_limit_is_rejected_before_reading_body(self):
        cookie = await self.open_ui()
        headers = (("content-length", str(U.MAX_REQUEST_BYTES + 1)),)
        denied = await self.request("/api/run/start", method="POST", body=MULTIPART, headers=(
            ("origin", ORIGIN), UI_HEADER, MULTIPART_HEADER, *headers,
        ))
        self.assertEqual(denied.status, 401)
        self.assertFalse(denied.body_read)
        rejected = await self.submit_form(cookie, MULTIPART, headers=headers)
        self.assertEqual(rejected.status, 413)
        self.assertFalse(rejected.body_read)
        self.assert_no_effects()

    async def test_stream_counts_full_body_and_ignores_understated_length(self):
        cookie = await self.open_ui()
        material = edit_form((b"A" * 100, b"B" * 100))
        chunks = tuple(material[start:start + 128] for start in range(0, len(material), 128))
        self.assertGreater(len(chunks), 3)
        with patch.object(U, "MAX_REQUEST_BYTES", 300):
            for headers in ((), (("content-length", "1"),)):
                with self.subTest(headers=headers):
                    response = await self.submit_form(cookie, chunks, headers=headers)
                    self.assertEqual(response.status, 413)
                    self.assertEqual(response.reads, 3)
        self.assert_no_effects()

    async def test_per_file_and_field_limits_apply_during_multipart_stream(self):
        cookie = await self.open_ui()
        payload = b"A" * 64
        body = edit_form((payload,))
        start = body.index(payload)
        chunks = (body[:start + 8], body[start + 8:start + 48], body[start + 48:])
        with patch.object(U, "MAX_REFERENCE_BYTES", 32), patch.object(U.Image, "open") as decoder:
            rejected = await self.submit_form(cookie, chunks)
            self.assertEqual(rejected.status, 413)
            self.assertEqual(rejected.reads, 2)
            decoder.assert_not_called()
        file_count = await self.submit_form(cookie, edit_form((b"fixture",) * (U.MAX_REFERENCES + 1)))
        self.assertEqual(file_count.status, 400)
        field_count = await self.submit_form(cookie, edit_form(fields=(("extra", "x"),) * U.MAX_FORM_FIELDS))
        self.assertEqual(field_count.status, 400)
        large_field = await self.submit_form(cookie, edit_form(fields=(("extra", "x" * (U.MAX_FIELD_BYTES + 1)),)))
        self.assertEqual(large_field.status, 413)
        self.assert_no_effects()

    async def test_incomplete_multipart_and_disconnected_upload_close_spools(self):
        cookie = await self.open_ui()
        partial = edit_form((image_bytes(),)).removesuffix(b"--fixture--\r\n")
        spools = []

        def track_spool(*args, **kwargs):
            file = tempfile.SpooledTemporaryFile(*args, **kwargs)
            spools.append(file)
            return file

        with (
            patch.object(U.ReferenceMultipartParser, "spool_max_size", 1),
            patch("starlette.formparsers.SpooledTemporaryFile", side_effect=track_spool),
        ):
            truncated = await self.submit_form(cookie, partial)
            disconnected = await self.submit_form(cookie, (partial, None))
        self.assertEqual(truncated.status, 400)
        self.assertEqual(disconnected.status, 400)
        self.assertEqual(len(spools), 2)
        self.assertTrue(all(file.closed for file in spools))
        self.assert_no_effects()

    async def test_ten_uploaded_references_keep_order_and_alpha(self):
        cookie = await self.open_ui()
        colors = [(value, 20, 30) if value % 2 else (value, 30, 40, 51) for value in range(10)]
        pictures = [image_bytes("RGBA" if len(color) == 4 else "RGB", color=color) for color in colors]
        self.use_worker_threads()
        body = edit_form(pictures)
        with patch.multiple(U, MAX_REQUEST_BYTES=len(body), MAX_REFERENCE_BYTES=max(map(len, pictures))):
            response = await self.submit_form(cookie, (body[:300], body[300:]), headers=(
                ("content-length", str(len(body))),
            ))
        self.assertEqual(response.status, 202, response.body)
        folder = self.jobs / json.loads(response.body)["job_dir"]
        inputs = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["jobs"][0]["inputs"]
        self.assertEqual(inputs, [f"inputs/ref_{index}.png" for index in range(1, 11)])
        for name, color in zip(inputs, colors, strict=True):
            with Image.open(folder / name) as png:
                self.assertEqual(png.getpixel((0, 0)), color)
        self.run_manifest.assert_called_once()

    async def test_urlencoded_t2i_keeps_form_compatibility_under_smaller_body_cap(self):
        cookie = await self.open_ui()
        self.use_worker_threads()
        headers = (cookie, ("origin", ORIGIN), UI_HEADER, ("content-type", "application/x-www-form-urlencoded"))
        with patch.object(U, "MAX_FORM_BYTES", 32):
            rejected = await self.request("/api/run/start", method="POST", headers=headers, body=b"prompt=" + b"a" * 32)
            self.assertEqual(rejected.status, 413)
            self.assert_no_effects()
            reply = await self.request("/api/run/start", method="POST", headers=headers, body=b"prompt=Synthetic+scene")
        self.assertEqual(reply.status, 202, reply.body)
        self.run_manifest.assert_called_once()

    async def test_invalid_form_fields_do_not_create_a_manifest_or_start_a_worker(self):
        cookie = await self.open_ui()
        headers = (cookie, ("origin", ORIGIN), UI_HEADER, ("content-type", "application/x-www-form-urlencoded"))
        for bad, field in (
            ({"seed": str(MAX_SEED + 1)}, "seed"), ({"seed": "-1"}, "seed"),
            ({"seed": "1.0"}, "種子碼"), ({"refine_prompt": "maybe"}, "refine_prompt"),
            ({"quant": "typo"}, "quant"), ({"pe_variant": "typo"}, "pe_variant"),
            ({"idle_minutes": "nan"}, "idle_minutes"),
        ):
            with self.subTest(bad=bad):
                response = await self.request(
                    "/api/run/start", method="POST", headers=headers,
                    body=urlencode({"prompt": "Synthetic scene", **bad}).encode(),
                )
                self.assertEqual(response.status, 400, response.body)
                self.assertIn(field, json.loads(response.body)["detail"])
                self.assert_no_effects()

    async def test_form_normalizes_aliases_and_preserves_full_unsigned_seed(self):
        cookie = await self.open_ui()
        self.use_worker_threads()
        headers = (cookie, ("origin", ORIGIN), UI_HEADER, ("content-type", "application/x-www-form-urlencoded"))
        response = await self.request(
            "/api/run/start", method="POST", headers=headers, body=urlencode({
                "prompt": "Synthetic scene", "seed": str(MAX_SEED), "quant": " q4_k_m ",
                "resolution": " 4mp ", "refine_prompt": "on", "pe_variant": "W4A8",
            }).encode(),
        )
        self.assertEqual(response.status, 202, response.body)
        reply = json.loads(response.body)
        self.assertEqual(reply["seed"], str(MAX_SEED))
        manifest = self.jobs / reply["job_dir"] / "manifest.json"
        job = json.loads(manifest.read_text())["jobs"][0]
        self.assertEqual((job["seed"], job["quant"], job["resolution"], job["pe_variant"]),
                         (MAX_SEED, "Q4_K_M", "2K", "w4a8"))
        self.assertIs(job["refine_prompt"], True)
        self.run_manifest.assert_called_once()

    async def test_edit_rejects_unknown_sizing_and_incomplete_reference_order(self):
        cookie = await self.open_ui()
        image = image_bytes()
        for files, fields, message in (
            ((image,), (("edit_sizing", "typo"),), "尺寸策略"),
            ((image,), (("ref_order", '{"kind":"upload"}'),), "ref_order"),
            ((image, image), (("ref_order", '[{"kind":"upload"}]'),), "ref_order"),
            ((image,), (("ref_order", '[{"kind":"upload"},{"kind":"upload"}]'),), "ref_order"),
        ):
            with self.subTest(fields=fields, files=len(files)):
                response = await self.submit_form(cookie, edit_form(files, fields=fields))
                self.assertEqual(response.status, 400, response.body)
                self.assertIn(message, json.loads(response.body)["detail"])
                self.assert_no_effects()

    async def test_empty_truncated_and_unsupported_images_leave_no_job(self):
        cookie = await self.open_ui()
        png = image_bytes(size=(64, 64), color=(12, 34, 56))
        for bad in (b"", png[:png.index(b"IDAT") + 7], image_bytes(format="PPM")):
            with self.subTest(size=len(bad)):
                response = await self.submit_form(cookie, edit_form((png, bad)))
                self.assertEqual(response.status, 400, response.body)
                self.assertIn("<image2>", json.loads(response.body)["detail"])
        self.assert_no_effects()

    async def test_dimensions_pixels_and_artifact_bytes_are_checked_before_pixels_load(self):
        cookie = await self.open_ui()
        cases = (
            ({"MAX_IMAGE_SIDE": 3}, image_bytes(size=(4, 1))),
            ({"MAX_IMAGE_PIXELS": 5}, image_bytes(size=(3, 2))),
        )
        for limits, png in cases:
            with self.subTest(limits=limits), patch.multiple(U, **limits), patch.object(Image.Image, "convert") as load:
                response = await self.submit_form(cookie, edit_form((png,)))
                self.assertEqual(response.status, 413)
                load.assert_not_called()
        for limits in ({"MAX_REFERENCE_BYTES": self.image.stat().st_size - 1}, {"MAX_IMAGE_PIXELS": 3}):
            with self.subTest(limits=limits), patch.multiple(U, **limits), patch.object(Image.Image, "convert") as load:
                response = await self.start_edit(cookie, ["fixture/out/sample.png"])
                self.assertEqual(response.status, 413)
                load.assert_not_called()
        self.assert_no_effects()

    async def test_only_one_upload_prepares_images_and_status_remains_responsive(self):
        cookie = await self.open_ui()
        body = edit_form((image_bytes(),))
        started = threading.Event()
        release = threading.Event()
        encode = U.write_reference_images

        def wait_and_encode(*args):
            started.set()
            if not release.wait(5):
                raise RuntimeError("encoder fixture timed out")
            encode(*args)

        self.use_worker_threads()
        with patch.object(U, "write_reference_images", side_effect=wait_and_encode):
            pending = asyncio.create_task(self.submit_form(cookie, body))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                status = await asyncio.wait_for(self.request("/api/run/status", headers=(cookie,)), timeout=2)
                self.assertEqual(status.status, 200)
                self.assertFalse(json.loads(status.body)["running"])
                self.assertFalse(pending.done())
                second = await self.submit_form(cookie, body)
                self.assertEqual(second.status, 409)
                self.assertFalse(second.body_read)
            finally:
                release.set()
                first = await pending
        self.assertEqual(first.status, 202, first.body)
        self.run_manifest.assert_called_once()

    async def test_cancelled_preparation_joins_encoder_and_removes_staging(self):
        cookie = await self.open_ui()
        body = edit_form((image_bytes(),))
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        encode = U.write_reference_images

        def wait_and_encode(*args):
            started.set()
            if not release.wait(5):
                raise RuntimeError("encoder fixture timed out")
            encode(*args)
            finished.set()

        with patch.object(U, "write_reference_images", side_effect=wait_and_encode) as encoder:
            pending = asyncio.create_task(self.submit_form(cookie, body))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                pending.cancel()
                await asyncio.sleep(0)
                pending.cancel()
                await asyncio.sleep(0)
                self.assertFalse(pending.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
        encoder.assert_called_once()
        self.assertTrue(finished.is_set())
        self.assertIsNone(self.state.generation)
        self.assert_no_effects()

    async def test_running_worker_retains_slot_after_submit_finishes(self):
        cookie = await self.open_ui()
        self.use_worker_threads(wait=False)
        started, release = threading.Event(), threading.Event()

        def wait_and_run(manifest, **_):
            self.assertTrue(manifest.is_file())
            started.set()
            self.assertTrue(release.wait(5))
            return {}

        self.run_manifest.side_effect = wait_and_run
        try:
            first = await self.submit_form(cookie, MULTIPART)
            self.assertEqual(first.status, 202, first.body)
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            second = await self.submit_form(cookie, edit_form((image_bytes(),)))
            self.assertEqual(second.status, 409)
            self.assertFalse(second.body_read)
        finally:
            release.set()
            await asyncio.to_thread(self.join_workers)
        self.assertIsNone(self.state.generation)
        self.assert_clean_staging()
        self.run_manifest.assert_called_once()
        self.assertEqual(len(list(self.jobs.glob("webui_*"))), 1)

    async def test_failed_thread_start_rolls_back_and_allows_retry(self):
        cookie = await self.open_ui()
        fail = Mock(side_effect=RuntimeError("synthetic credential value"))
        with patch.object(W, "threading", SimpleNamespace(Thread=lambda **_: SimpleNamespace(start=fail))):
            with self.assertLogs(W.LOGGER, level="WARNING") as captured:
                rejected = await self.submit_form(cookie, edit_form((image_bytes(),)))
        self.assertEqual(rejected.status, 503, rejected.body)
        self.assertNotIn(b"synthetic credential value", rejected.body)
        self.assertNotIn("synthetic credential value", "\n".join(captured.output))
        fail.assert_called_once()
        self.assertIsNone(self.state.generation)
        self.assert_no_effects()
        self.use_worker_threads()
        self.assertEqual((await self.submit_form(cookie, MULTIPART)).status, 202)
        self.assert_clean_staging()
        self.run_manifest.assert_called_once()

    async def test_failed_publication_does_not_let_waiting_worker_run(self):
        cookie = await self.open_ui()
        self.use_worker_threads()
        with patch.object(Path, "rename", side_effect=PermissionError("fixture rename failed")):
            failed = await self.submit_form(cookie, MULTIPART)
        self.assertEqual(failed.status, 400, failed.body)
        self.assertEqual(len(self.worker_threads), 1)
        self.assertIsNone(self.state.generation)
        self.assert_no_effects()

    async def test_targeted_preparation_cancel_waits_for_encoder_and_leaves_slot_reusable(self):
        cookie = await self.open_ui()
        body = edit_form((image_bytes(),))
        started, release = threading.Event(), threading.Event()
        encode = U.write_reference_images

        def wait_and_encode(*args):
            started.set()
            self.assertTrue(release.wait(5))
            encode(*args)

        async def cancel(preparation_id):
            return await self.request("/api/colab/cancel", method="POST",
                                      headers=(cookie, ("origin", ORIGIN), UI_HEADER, JSON_HEADER),
                                      body=json.dumps({"preparation_id": preparation_id}).encode())

        with patch.object(U, "write_reference_images", side_effect=wait_and_encode):
            pending = asyncio.create_task(self.submit_form(cookie, body))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                response = await self.request("/api/run/status", headers=(cookie,))
                status = json.loads(response.body)
                self.assertTrue(status["preparing"])
                self.assertTrue(status["can_cancel"])
                self.assertEqual((await cancel("older-attempt")).status, 409)
                accepted = await cancel(status["preparation_id"])
                self.assertEqual(accepted.status, 202)
                self.assertFalse(pending.done())
                self.assertEqual((await self.submit_form(cookie, MULTIPART)).status, 409)
            finally:
                release.set()
                rejected = await asyncio.wait_for(pending, timeout=3)
        self.assertEqual(rejected.status, 409, rejected.body)
        self.assertIsNone(self.state.generation)
        self.assert_clean_staging()
        self.run_manifest.assert_not_called()
        self.cloud["request_cancel"].assert_not_called()
        self.assertEqual(list(self.jobs.glob("webui_*")), [])
        self.use_worker_threads()
        self.assertEqual((await self.submit_form(cookie, MULTIPART)).status, 202)

    async def test_lifespan_shutdown_interrupts_upload_and_rejects_new_work(self):
        cookie = await self.open_ui()
        blocked = asyncio.Event()
        body = edit_form((image_bytes(),))
        async with W.app.router.lifespan_context(W.app):
            async def pause(reads):
                if reads:
                    blocked.set()
                    await asyncio.Future()

            pending = asyncio.create_task(self.submit_form(cookie, (body[:-8], body[-8:]), before_read=pause))
            await asyncio.wait_for(blocked.wait(), timeout=3)
        self.assertEqual((await asyncio.wait_for(pending, timeout=3)).status, 409)
        rejected = await self.submit_form(cookie, MULTIPART)
        self.assertEqual(rejected.status, 503)
        self.assertFalse(rejected.body_read)
        self.assertIsNone(self.state.generation)
        self.assert_no_effects()

    async def test_lifespan_shutdown_signals_worker_and_keeps_published_job(self):
        cookie = await self.open_ui()
        self.use_worker_threads(wait=False)
        started, allow_finish = threading.Event(), threading.Event()
        signal = []

        def fake_run(manifest, *, should_abort, **_):
            self.assertTrue(manifest.is_file())
            started.set()
            self.assertTrue(allow_finish.wait(5))
            signal.append(should_abort())
            raise W.R.RunCancelled("fixture batch cancelled")

        self.run_manifest.side_effect = fake_run
        lifespan = W.app.router.lifespan_context(W.app)
        await lifespan.__aenter__()
        try:
            response = await self.submit_form(cookie, MULTIPART)
            self.assertEqual(response.status, 202)
            job_dir = self.jobs / json.loads(response.body)["job_dir"]
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
        finally:
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            try:
                await asyncio.sleep(0)
                self.assertTrue(self.state.closing)
                self.assertFalse(shutdown.done())
                self.assertEqual((await self.submit_form(cookie, MULTIPART)).status, 503)
            finally:
                allow_finish.set()
                await asyncio.wait_for(shutdown, timeout=3)
        self.assertEqual(signal, ["cancel"])
        self.assertTrue((job_dir / "manifest.json").is_file())
        self.assertIsNone(self.state.generation)
        self.cloud["release_runtime"].assert_not_called()
        self.cloud["sessions"].assert_not_called()

    async def test_stop_all_during_preflight_reports_local_cancellation_pending(self):
        cookie = await self.open_ui()
        self.use_worker_threads(wait=False)
        started, allow_finish = threading.Event(), threading.Event()

        def fake_run(_, *, should_abort, **__):
            started.set()
            self.assertTrue(allow_finish.wait(5))
            self.assertEqual(should_abort(), "cancel")
            raise W.R.RunCancelled("fixture batch cancelled")

        self.run_manifest.side_effect = fake_run
        try:
            self.assertEqual((await self.submit_form(cookie, MULTIPART)).status, 202)
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            stop = await self.request("/api/colab/stop", method="POST", body=b'{"all":true}',
                                      headers=(cookie, ("origin", ORIGIN), UI_HEADER, JSON_HEADER))
            self.assertEqual(stop.status, 202, stop.body)
            data = json.loads(stop.body)
            self.assertTrue(data["generation_cancel_pending"])
            self.assertFalse(data["ok"])
            self.assertEqual(data["verified_gone"], [])
        finally:
            allow_finish.set()
            await asyncio.to_thread(self.join_workers)
        self.assertEqual(self.state.stage, "cancelled")
        self.cloud["release_runtime"].assert_not_called()

    async def test_lifespan_restart_only_removes_abandoned_managed_staging(self):
        await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", """
import os
import sys
from pathlib import Path
from cstudio.web_inputs import generation_staging
with generation_staging(Path(sys.argv[1])) as stage:
    (stage / 'partial.png').write_bytes(b'fixture temporary data')
    os._exit(0)
""", str(self.jobs)],
            check=True, timeout=5,
        )
        root = self.jobs / U.STAGING_DIR
        abandoned, = root.glob("pending-*")
        unrelated = root / "pending-keepthis"
        unrelated.mkdir()
        legacy = self.jobs / ".webui-old"
        legacy.mkdir()
        self.state.begin_shutdown()
        with U.generation_staging(self.jobs) as active:
            (active / "copy.png").write_bytes(b"other process fixture")
            async with W.app.router.lifespan_context(W.app):
                self.assertFalse(abandoned.exists())
                self.assertEqual((active / "copy.png").read_bytes(), b"other process fixture")
                self.assertFalse(self.state.closing)
        self.assertTrue(legacy.is_dir())
        self.assertTrue(self.image.is_file())
        self.assertEqual(list(root.glob("pending-*")), [unrelated])

    async def test_shutdown_timeout_leaves_an_inflight_worker_and_its_keep_intact(self):
        cookie = await self.open_ui()
        self.use_worker_threads(wait=False)
        started, allow_finish = threading.Event(), threading.Event()
        W.C.KEEP.mkdir()
        keep = W.C.KEEP / "example.json"
        lease = b'{"session":"example","created_by":"owned","account_scope":"fixture-account"}'
        keep.write_bytes(lease)

        def fake_run(_, *, should_abort, **__):
            started.set()
            self.assertTrue(allow_finish.wait(5))
            self.assertEqual(should_abort(), "cancel")
            raise W.R.RunCancelled("fixture batch cancelled")

        self.run_manifest.side_effect = fake_run
        try:
            with patch.object(W, "WORKER_DRAIN_SECONDS", 0), self.assertLogs(W.LOGGER, level="WARNING"):
                async with W.app.router.lifespan_context(W.app):
                    self.assertEqual((await self.submit_form(cookie, MULTIPART)).status, 202)
                    self.assertTrue(await asyncio.to_thread(started.wait, 3))
            self.assertTrue(self.state.running)
            self.assertIsNotNone(self.state.generation)
            self.assertEqual(keep.read_bytes(), lease)
            self.cloud["release_runtime"].assert_not_called()
        finally:
            allow_finish.set()
            await asyncio.to_thread(self.join_workers)
        self.assertIsNone(self.state.generation)
        self.assertEqual(self.state.stage, "cancelled")
        self.assertEqual(keep.read_bytes(), lease)

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
                name, _, filename = artifact.split("/")
                thumbnail = await self.request(f"/api/gallery/thumbnail/{name}/{filename}", headers=(cookie,))
                self.assertIn(thumbnail.status, (400, 404))
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

    async def test_gallery_api_pages_and_serves_thumb_with_original_reference_intact(self):
        cookie = await self.open_ui()
        Image.new("RGBA", (384, 192), (200, 20, 30, 80)).save(self.image)
        second = self.image.with_name("older.png")
        Image.new("RGB", (4, 3)).save(second)
        os.utime(second, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        newest = await self.request("/api/gallery?limit=1", headers=(cookie,))
        data = json.loads(newest.body)
        item, = data["items"]
        self.assertEqual(item["filename"], "sample.png")
        thumb = await self.request(item["thumbnail_url"], headers=(cookie,))
        self.assertEqual(thumb.status, 200)
        self.assertEqual(thumb.headers[b"content-type"], b"image/png")
        self.assertEqual(thumb.headers[b"cache-control"], b"no-store")
        self.assertEqual(thumb.headers[b"cross-origin-resource-policy"], b"same-origin")
        with Image.open(io.BytesIO(thumb.body)) as preview:
            self.assertEqual(preview.size, (192, 96))
            self.assertEqual(preview.getpixel((0, 0))[3], 80)
        page_two = await self.request("/api/gallery?" + urlencode({
            "limit": 1, "before": data["next_cursor"],
        }), headers=(cookie,))
        self.assertEqual(page_two.status, 200)
        other = json.loads(page_two.body)
        self.assertNotEqual(other["items"][0]["artifact_id"], item["artifact_id"])
        self.assertIsNone(other["next_cursor"])
        original = await self.request(item["url"], headers=(cookie,))
        self.assertEqual(original.body, (self.image.parent / item["filename"]).read_bytes())
        invalid = await self.request("/api/gallery?" + urlencode({"before": "10:../out/bad.png"}), headers=(cookie,))
        self.assertEqual(invalid.status, 400)
        self.assertEqual(sorted(path.name for path in self.image.parent.iterdir()), ["older.png", "sample.png"])

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
        self.use_worker_threads()
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
        self.use_worker_threads()
        for path, extra, body in CONTROLS:
            with self.subTest(path=path):
                response = await self.request(path, method="POST", body=body, headers=(
                    cookie, ("origin", ORIGIN), ("sec-fetch-site", "same-origin"), UI_HEADER, *extra,
                ))
                self.assertEqual(response.status, 202 if path == "/api/run/start" else 200, response.body)
                self.assertTrue(json.loads(response.body)["ok"])
                if path == "/api/colab/login/code":
                    self.assertEqual(json.loads(response.body), {"ok": True})
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

    async def test_web_login_requires_explicit_state_and_never_reflects_codes_or_provider_errors(self):
        cookie = await self.open_ui()
        headers = (cookie, ("origin", ORIGIN), UI_HEADER, JSON_HEADER)
        code = "fixture-auth-code"
        for payload in (
            [code], {"code": code}, {"code": code, "state": None}, {"code": code, "state": ""},
            {"code": code, "state": {"unexpected": True}},
        ):
            with self.subTest(payload_type=type(payload).__name__):
                response = await self.request("/api/colab/login/code", method="POST",
                                              body=json.dumps(payload).encode(), headers=headers)
                self.assertEqual(response.status, 400)
                self.assertNotIn(code.encode(), response.body)
        self.assert_no_effects()

        private = (code, "fixture-access-token", "fixture-refresh-token", "fixture-pkce-verifier", str(self.token))
        self.finish_login.side_effect = RuntimeError(" ".join(private))
        response = await self.request("/api/colab/login/code", method="POST", headers=headers,
                                      body=json.dumps({"code": code, "state": "exact-state"}).encode())
        self.assertEqual(response.status, 400)
        self.finish_login.assert_called_once_with(code, state="exact-state")
        self.cloud["usage"].assert_not_called()
        for value in private:
            self.assertNotIn(value.encode(), response.body)

        self.begin_login.side_effect = RuntimeError(" ".join(private))
        response = await self.request("/api/colab/login/url", method="POST", headers=headers)
        self.assertEqual(response.status, 500)
        for value in private:
            self.assertNotIn(value.encode(), response.body)
        self.assertEqual(self.token.read_text(encoding="utf-8"), "original token")

    async def test_diagnostic_status_masks_oauth_and_runtime_fields(self):
        cookie = await self.open_ui()
        self.cloud["usage"].return_value = (10.0, "10 compute units Authorization: Bearer fixture-ui-access")
        self.cloud["sessions"].return_value = ([], "x-colab-runtime-proxy-token: fixture-ui-proxy")
        slow = await self.request("/api/colab/status?force=true", headers=(cookie, UI_HEADER))
        self.assertEqual(slow.status, 200)
        self.assertEqual(json.loads(slow.body)["cu"], 10.0)
        self.assertIn(b"compute units", slow.body)
        self.assertNotIn(b"fixture-ui-access", slow.body)
        self.assertNotIn(b"fixture-ui-proxy", slow.body)
        self.state.on_runner_progress({
            "stage": "error",
            "log": ["token refresh failed, refresh_token=fixture-ui-refresh"],
            "error": "could not connect to https://example.test/runtime?token=fixture-ui-runtime",
        })
        fast = await self.request("/api/run/status", headers=(cookie,))
        self.assertEqual(fast.status, 200)
        self.assertNotIn(b"fixture-ui-refresh", fast.body)
        self.assertNotIn(b"fixture-ui-runtime", fast.body)
        self.assertIn(b"[redacted]", fast.body)

    async def test_status_routes_share_session_view_and_reject_expired_browser_selection(self):
        cookie = await self.open_ui()
        self.cloud["sessions"].return_value = (["external"], "[external] fixture-old-endpoint | Hardware: A100")
        slow = await self.request("/api/colab/status?force=true", headers=(cookie, UI_HEADER))
        fast = await self.request("/api/run/status", headers=(cookie,))
        slow_view, fast_view = json.loads(slow.body), json.loads(fast.body)
        self.assertEqual(slow_view["view_id"], fast_view["view_id"])
        self.assertGreater(fast_view["revision"], slow_view["revision"])
        self.assertEqual(fast_view["sessions"], slow_view["sessions"])
        self.cloud["sessions"].assert_called_once()
        self.assertNotIn(b"fixture-old-endpoint", slow.body)
        selection = fast_view["sessions"][0]["id"]
        self.cloud["sessions"].return_value = (["external"], "[external] fixture-new-endpoint | Hardware: L4")
        await self.request("/api/colab/status?force=true", headers=(cookie, UI_HEADER))

        form_header = ("content-type", "application/x-www-form-urlencoded")
        origin = ("origin", ORIGIN)
        form = urlencode({"prompt": "Synthetic prompt", "session": "external", "session_id": selection}).encode()
        start = await self.request("/api/run/start", method="POST", headers=(cookie, UI_HEADER, origin, form_header),
                                   body=form)
        self.assertEqual(start.status, 409)
        self.run_manifest.assert_not_called()
        self.assertEqual([path for path in self.jobs.iterdir() if path.name != U.STAGING_DIR],
                         [self.image.parents[1]])
        self.assert_clean_staging()
        for action in ("keep", "cancel"):
            rejected = await self.request(f"/api/colab/{action}", method="POST",
                                          headers=(cookie, UI_HEADER, origin, JSON_HEADER),
                                          body=json.dumps({"session": "external", "session_id": selection}).encode())
            self.assertEqual(rejected.status, 409)
        self.cloud["keep_target"].assert_not_called()
        self.cloud["request_cancel"].assert_not_called()

        current = await self.request("/api/run/status", headers=(cookie,))
        new_id = json.loads(current.body)["sessions"][0]["id"]
        form = urlencode({"prompt": "Synthetic prompt", "session": "external", "session_id": new_id}).encode()
        self.use_worker_threads()
        accepted = await self.request("/api/run/start", method="POST",
                                      headers=(cookie, UI_HEADER, origin, form_header), body=form)
        self.assertEqual(accepted.status, 202, accepted.body)
        self.assertEqual(self.run_manifest.call_args.kwargs["expected_endpoint"], "fixture-new-endpoint")
        finished = await self.request("/api/run/status", headers=(cookie,))
        self.assertFalse(json.loads(finished.body)["running"])
        self.assertEqual(self.cloud["sessions"].call_count, 3, "completion triggers one additional listing")

    def test_launcher_binds_only_supported_loopback_hosts_and_disables_forwarding(self):
        with patch("uvicorn.Server.run", autospec=True) as run, patch.object(W.webbrowser, "open") as open_browser, patch("builtins.print"):
            for host in ("0.0.0.0", "::", "192.0.2.10", "workbench.example.test"):
                with self.subTest(host=host), self.assertRaisesRegex(SystemExit, "loopback|只允許"):
                    W.launch_ui(host=host, open_browser=True)
            run.assert_not_called()
            open_browser.assert_not_called()
            for host, expected in (("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"), ("::1", "::1")):
                with self.subTest(host=host):
                    W.launch_ui(host=host, port=8765)
                    config = run.call_args.args[0].config
                    self.assertIs(config.app, W.app)
                    self.assertEqual(config.host, expected)
                    self.assertEqual(config.port, 8765)
                    self.assertEqual(config.log_level, "info")
                    self.assertFalse(config.proxy_headers)
                    self.assertEqual(config.timeout_graceful_shutdown, W.REQUEST_DRAIN_SECONDS)
