"""File boundaries shared by the local client and the Colab runner."""
import hashlib
import json
import re
import stat
from pathlib import Path


JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,60}")
RUN_ID = re.compile(r"[0-9a-f]{12}")
SHA256 = re.compile(r"[0-9a-f]{64}")
INVALID_PATH_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f\x7f]')
WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³])", re.IGNORECASE)


def relative_parts(value: str) -> list[str]:
    """Parse a canonical path that is safe on both POSIX and Windows."""
    if not isinstance(value, str) or not value or value.startswith("~"):
        raise ValueError("需要使用以 / 分隔的相對檔案路徑")
    parts = value.split("/")
    if any(
        not part or part.endswith((".", " "))
        or INVALID_PATH_CHARS.search(part)
        or WINDOWS_DEVICE.fullmatch(part.partition(".")[0].rstrip())
        for part in parts
    ):
        raise ValueError(f"不合法的相對檔案路徑：{value!r}")
    return parts


def confined_path(root: Path, relative: str, *, directory: bool = False, missing_ok: bool = False) -> Path:
    """Check each component without following links. The root's ancestors are trusted."""
    parts = relative_parts(relative)
    current = root
    for index, part in enumerate(["", *parts]):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing_ok and index:
                return root.joinpath(*parts)
            raise
        expected = stat.S_ISDIR if index < len(parts) or directory else stat.S_ISREG
        if not expected(info.st_mode) or getattr(info, "st_reparse_tag", 0):
            raise ValueError(f"路徑必須是一般檔案或目錄，且不可使用符號連結：{current}")
    return current


def bundle_digest(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
