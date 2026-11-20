"""Mask credentials in diagnostics without changing the surrounding syntax."""
import json
import re
from urllib.parse import quote, quote_plus

REDACTED = "[redacted]"
_KEY = (
    r"authorization|code|(?:auth|authorization|oauth)[_-]?code|code[_-]?verifier|client[_-]?secret|"
    r"(?:access|refresh|id|oauth)[_-]?token|"
    r"(?:x[_-])?(?:colab[_-])?(?:(?:runtime[_-]?)?proxy|xsrf)[_-]?token|token"
)
_ASSIGNMENT = re.compile(
    rf"""(?P<head>(?<![\w-])["']?(?:{_KEY})["']?[ \t]*[:=][ \t]*)
    (?P<value>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|
    (?:Bearer|Basic)[ \t]+[\w.~+/=-]+|[^\s,;&"'{{}}\[\]()<>]+)""",
    re.IGNORECASE | re.VERBOSE,
)
_BEARER = re.compile(r"\b(Bearer[ \t]+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SCALAR = re.compile(r"null|None|true|false|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


def redact(text: str, secrets=(), *, labels: bool = True) -> str:
    variants = set()
    for secret in secrets:
        if isinstance(secret, str) and secret:
            variants.update((secret, quote(secret, safe=""), quote_plus(secret), json.dumps(secret)[1:-1]))
    for variant in sorted(variants, key=len, reverse=True):
        text = text.replace(variant, REDACTED)
    if not labels:
        return text

    def replace(match):
        value = match["value"]
        if value[0] in ("'", '"'):
            return match["head"] + value[0] + REDACTED + value[0]
        if match["head"][0] in ("'", '"') and _SCALAR.fullmatch(value):
            return match[0]
        return match["head"] + REDACTED

    return _BEARER.sub(r"\1" + REDACTED, _ASSIGNMENT.sub(replace, text))


def redact_diagnostics(record: dict) -> dict:
    safe = {
        key: REDACTED if isinstance(value, str) and re.fullmatch(_KEY, key, re.I) else value
        for key, value in record.items()
    }
    for field in ("error", "traceback"):
        if isinstance(safe.get(field), str):
            safe[field] = redact(safe[field])
    if isinstance(safe.get("log"), list):
        safe["log"] = [redact(str(line)) for line in safe["log"]]
    return safe
