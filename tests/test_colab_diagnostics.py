"""Check diagnostic redaction against synthetic Colab protocol and transport output."""
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote

from cstudio import cli, colab_client as C
from cstudio.redact import REDACTED, redact


class ColabDiagnosticTests(unittest.TestCase):
    def test_check_reports_the_scoped_interpreters_patch_status(self):
        status = {"version": "0.7.4", "package": "fixture/site-packages/colab_cli",
                  "files": {"client.py": "patched", "console.py": "legacy"}}
        with (
            patch.object(cli, "patch_status", return_value=(status, 1)),
            patch.object(C, "token_ok", return_value=False),
            patch.object(C, "usage") as usage,
            patch.object(C, "sessions") as sessions,
            redirect_stdout(io.StringIO()) as out,
            self.assertRaises(SystemExit) as result,
        ):
            cli.cmd_check(None)
        self.assertEqual(result.exception.code, 1)
        self.assertIn("0.7.4", out.getvalue())
        self.assertIn(status["package"], out.getvalue())
        self.assertIn("client.py=patched, console.py=legacy", out.getvalue())
        usage.assert_not_called()
        sessions.assert_not_called()

    def test_masks_headers_urls_serialized_values_and_known_secrets(self):
        payload = {
            "Authorization": "Basic fixture-http-auth",
            "accessToken": "fixture-access",
            "refresh_token": "fixture-refresh",
            "code_verifier": "fixture-verifier",
            "X-Colab-Runtime-Proxy-Token": "fixture-runtime",
            "code": 412,
        }
        self.assertEqual(json.loads(redact(json.dumps(payload))), dict.fromkeys(
            (key for key in payload if key != "code"), REDACTED,
        ) | {"code": 412})
        url = "https://example.test/api?code=fixture%2Fcode&name=normal&token=fixture%2Btoken"
        self.assertEqual(redact(url), f"https://example.test/api?code={REDACTED}&name=normal&token={REDACTED}")
        self.assertEqual(redact("server header='Bearer fixture-auth'"), f"server header='Bearer {REDACTED}'")
        secret = "fixture/known+secret"
        escaped = quote(secret, safe="")
        self.assertEqual(redact(f"bare: {secret}, encoded: {escaped}", (secret,)),
                         f"bare: {REDACTED}, encoded: {REDACTED}")

    def test_failed_transport_omits_raw_output_but_preserves_allocation_refusal(self):
        completed = SimpleNamespace(
            returncode=1, stdout="fixture-unlabelled-secret",
            stderr=f"{C.ALLOCATION_REFUSED} upstream includes fixture-proxy-token",
        )
        with patch.object(C.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "colab download 失敗") as transfer:
                C.colab("download", "local-fixture", account="fixture-scope")
            with self.assertRaisesRegex(RuntimeError, "colab new 失敗") as allocation:
                C.colab("new", account="fixture-scope")
            self.assertEqual(C.colab("usage", check=False, account="fixture-scope"), "")
        self.assertIn(C.ALLOCATION_REFUSED, str(allocation.exception).splitlines())
        for error in (transfer.exception, allocation.exception):
            self.assertNotIn("fixture-unlabelled-secret", str(error))
            self.assertNotIn("fixture-proxy-token", str(error))

    def test_successful_status_protocol_keeps_identity_and_masks_public_diagnostics(self):
        listing = "[studio] runtime-endpoint | Hardware: L4 | Shape: Standard | Variant: GPU"
        completed = SimpleNamespace(returncode=0, stdout=listing, stderr="")
        with patch.object(C.subprocess, "run", return_value=completed):
            self.assertEqual(C.listed_assignments(C.colab("sessions", account="fixture-scope")), {
                "runtime-endpoint": "studio",
            })
            completed.stdout = "10 compute units\nAuthorization: Bearer fixture-passed-access"
            units, summary = C.usage(account="fixture-scope")
            prompt = "A poster with the literal sample token=example and CSTUDIO_EXEC_OK printed on it"
            completed.stdout = "CSTUDIO_STATUS " + json.dumps({"run_id": "fixture-run", "refined_prompt": prompt}) + "\nCSTUDIO_EXEC_OK"
            completed.stderr = "Authorization: Bearer fixture-stderr"
            completed.returncode = 1
            status = C.poll_remote_status("studio", account="fixture-scope", endpoint="runtime-endpoint")
        self.assertEqual(units, 10)
        self.assertNotIn("fixture-passed-access", summary)
        self.assertIn("10 compute units", summary)
        self.assertEqual(status, {"run_id": "fixture-run", "refined_prompt": prompt})
