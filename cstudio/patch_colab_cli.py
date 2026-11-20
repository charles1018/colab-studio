"""Verify and patch google-colab-cli in the Python environment running Studio."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import stat
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path

DIST = "google-colab-cli"
VERSION = "0.7.4"

# Pristine file hashes from the wheel verified against uv.lock. Normalize newlines
# when reading so installs patched by the old Windows script can be recognized.
SOURCE_HASHES = {
    "client.py": "af6cb28c96ac9dc3d1d1a0a7587ad87acc8e0eed36b63de301ce4b9f026755b1",
    "console.py": "d2c9c1e8c42bfa0107ef1fcff3ba4f572581b25484248642b49b88c754bf2c60",
}
CLIENT_ANCHOR = "        response = self.session.request(\n            method, endpoint,"
CLIENT_FIX = (
    '        if str(method).upper() == "POST" and "data" not in kwargs and "json" not in kwargs:  # cstudio: 411 fix\n'
    '            kwargs["json"] = {}\n'
)
LEGACY_CLIENT_FIX = CLIENT_FIX.replace("cstudio: 411 fix", "h3skill: 411 fix")
CONSOLE_IMPORTS = "import termios\nimport threading\nimport time\nimport tty\n"
CONSOLE_IMPORT_FIX = (
    "try:\n    import termios\nexcept ImportError:  # Windows\n    termios = None\n"
    "import threading\nimport time\n"
    "try:\n    import tty\nexcept ImportError:\n    tty = None\n"
)
CONSOLE_TTY = "    is_tty = sys.stdin.isatty()\n    fd = sys.stdin.fileno() if is_tty else None\n"
CONSOLE_TTY_FIX = (
    "    is_tty = sys.stdin.isatty()\n"
    '    if is_tty and (termios is None or tty is None or not hasattr(signal, "SIGWINCH")):\n'
    '        raise RuntimeError("Interactive Colab console requires a POSIX terminal")\n'
    "    fd = sys.stdin.fileno() if is_tty else None\n"
)


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FilePatch:
    state: str
    source: str = ""
    replacement: str | None = None


def _package_path() -> Path:
    try:
        dist = importlib.metadata.distribution(DIST)
    except importlib.metadata.PackageNotFoundError:
        raise PatchError(f"找不到 {DIST}，請先執行 `uv sync --locked`") from None
    if dist.version != VERSION:
        raise PatchError(f"{DIST} {dist.version} 版本不符，需要 {VERSION}")
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve():
        raise PatchError("請從 Studio 的虛擬環境執行")
    spec = importlib.util.find_spec("colab_cli")
    if spec is None or spec.origin is not None or spec.submodule_search_locations is None:
        raise PatchError("目前的 Python 找不到預期的 colab_cli namespace 套件")
    locations = {Path(path).resolve() for path in spec.submodule_search_locations}
    package = Path(dist.locate_file("colab_cli")).resolve()
    libraries = {Path(sysconfig.get_path(kind)).resolve() for kind in ("purelib", "platlib")}
    if locations != {package} or package.parent not in libraries or not package.is_relative_to(prefix):
        raise PatchError("colab_cli 的匯入位置與目前虛擬環境的套件不符")
    return package


def _inspect_file(path: Path) -> FilePatch:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return FilePatch("incompatible")
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return FilePatch("incompatible")

    if path.name == "client.py":
        cstudio = CLIENT_FIX + CLIENT_ANCHOR
        legacy = LEGACY_CLIENT_FIX + CLIENT_ANCHOR
        pristine = source.replace(cstudio, CLIENT_ANCHOR, 1).replace(legacy, CLIENT_ANCHOR, 1)
        if _matches_pristine(path.name, pristine) and pristine.count(CLIENT_ANCHOR) == 1:
            replacement = pristine.replace(CLIENT_ANCHOR, cstudio, 1)
            if source == replacement or source == pristine.replace(CLIENT_ANCHOR, legacy, 1):
                return FilePatch("patched")
            if source == pristine:
                return FilePatch("unpatched", source, replacement)
    else:
        pristine = source.replace(CONSOLE_TTY_FIX, CONSOLE_TTY, 1).replace(CONSOLE_IMPORT_FIX, CONSOLE_IMPORTS, 1)
        if (_matches_pristine(path.name, pristine) and pristine.count(CONSOLE_IMPORTS) == 1
                and pristine.count(CONSOLE_TTY) == 1):
            legacy = pristine.replace(CONSOLE_IMPORTS, CONSOLE_IMPORT_FIX, 1)
            replacement = legacy.replace(CONSOLE_TTY, CONSOLE_TTY_FIX, 1)
            if source == replacement:
                return FilePatch("patched")
            if source == legacy:
                return FilePatch("legacy", source, replacement)
            if source == pristine:
                return FilePatch("unpatched", source, replacement)
    return FilePatch("incompatible")


def _matches_pristine(name: str, source: str) -> bool:
    return hashlib.sha256(source.encode("utf-8")).hexdigest() == SOURCE_HASHES[name]


def _install_patch(path: Path, change: FilePatch) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}-", encoding="utf-8", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(change.replacement)
        # uv can hardlink installed files to its cache. Replace the local entry
        # so the cache and any other environment keep their original contents.
        if not stat.S_ISREG(path.lstat().st_mode) or path.read_text(encoding="utf-8") != change.source:
            raise PatchError(f"{path.name} 在套用前已變更，請重新查詢狀態")
        temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def patch_status() -> tuple[dict, int]:
    """Return a read-only status and the corresponding command exit code."""
    try:
        package = _package_path()
    except (PatchError, OSError, ValueError) as exc:
        return {"error": str(exc)}, 2
    files = {name: _inspect_file(package / name).state for name in SOURCE_HASHES}
    code = 2 if "incompatible" in files.values() else int(any(state != "patched" for state in files.values()))
    return {"version": VERSION, "package": str(package), "files": files}, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="只讀取 JSON 狀態（0 已補、1 待補、2 不相容）")
    if parser.parse_args(argv).status:
        status, code = patch_status()
        print(json.dumps(status, ensure_ascii=False))
        sys.exit(code)

    try:
        package = _package_path()
        changes = {name: _inspect_file(package / name) for name in SOURCE_HASHES}
        incompatible = [name for name, change in changes.items() if change.state == "incompatible"]
        if incompatible:
            raise PatchError(f"來源不符或無法讀取：{', '.join(incompatible)}")
        for name, change in changes.items():
            if change.replacement is not None:
                _install_patch(package / name, change)
            print(f"{'已補丁' if change.replacement is not None else '已經補過'}：{package / name}")
    except (PatchError, OSError, ValueError) as exc:
        print(f"補丁失敗：{exc}。可用 `cstudio-patch --status` 查詢各檔案。", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
