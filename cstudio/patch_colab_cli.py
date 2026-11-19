# -*- coding: utf-8 -*-
"""Fixes for google-colab-cli 0.7.4 inside the uv project .venv. Safe to run repeatedly.
1. console.py imports termios/tty (POSIX only) at module level -> wrapped in try/except.
2. client.py sends body-less POSTs (e.g. GPU assign) -> sends empty JSON body `{}` to avoid HTTP 411 Length Required.
3. Disables colab-cli's startup update check in ~/.config/colab-cli/settings.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CLIENT_ANCHOR = "        response = self.session.request(\n            method, endpoint,"
CLIENT_MARK = "cstudio: 411 fix"
CLIENT_FIX = (
    '        if str(method).upper() == "POST" and "data" not in kwargs and "json" not in kwargs:  # '
    + CLIENT_MARK
    + "\n"
    '            kwargs["json"] = {}\n'
)


def candidate_roots():
    roots = [ROOT / ".venv"]
    return [r for r in roots if r.exists()]


def patch_console(p: Path) -> str:
    s = p.read_text(encoding="utf-8")
    if "termios = None" in s:
        return "已經補過"
    s = s.replace("import termios\n", "try:\n    import termios\nexcept ImportError:  # Windows\n    termios = None\n", 1)
    s = s.replace("import tty\n", "try:\n    import tty\nexcept ImportError:\n    tty = None\n", 1)
    p.write_text(s, encoding="utf-8")
    return "已補丁"


def patch_client(p: Path) -> str:
    s = p.read_text(encoding="utf-8")
    if CLIENT_MARK in s or "h3skill: 411 fix" in s:
        return "已經補過"
    if s.count(CLIENT_ANCHOR) != 1:
        return "⚠️ client.py 版本結構不同，略過"
    p.write_text(s.replace(CLIENT_ANCHOR, CLIENT_FIX + CLIENT_ANCHOR, 1), encoding="utf-8")
    return "已補丁"


def main():
    conf = Path.home() / ".config" / "colab-cli"
    conf.mkdir(parents=True, exist_ok=True)
    settings = conf / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    data["enable_update_check"] = False
    settings.write_text(json.dumps(data), encoding="utf-8")

    pkgs = []
    for r in candidate_roots():
        pkgs.extend(r.glob("**/colab_cli"))
    if not pkgs:
        sys.exit("找不到 colab_cli：請先執行 `uv sync`")
    for pkg in pkgs:
        for name, fn in (("console.py", patch_console), ("client.py", patch_client)):
            if (pkg / name).exists():
                print(f"{fn(pkg / name)}：{pkg / name}")


if __name__ == "__main__":
    main()
