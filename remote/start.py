# -*- coding: utf-8 -*-
"""Runs ON the Colab VM through `colab exec`: starts remote_runner.py in the background and returns at once."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

JOB = Path("/content/cstudio")
status = JOB / "status.json"
out_dir = JOB / "out"

running = subprocess.run(["pgrep", "-f", "cstudio/remote_runner.py"], capture_output=True, text=True).stdout.strip()
if running:
    # Check if the already-running process is for the same _run_id (idempotent retry after a lost exec reply)
    same_run = False
    try:
        req_run = json.loads((JOB / "manifest.json").read_text(encoding="utf-8")).get("_run_id")
        cur_run = json.loads(status.read_text(encoding="utf-8")).get("run_id") if status.exists() else None
        same_run = bool(req_run and req_run == cur_run)
    except Exception:
        pass
    print(json.dumps({"started": same_run, "already_running": True, "same_run": same_run, "pid": running}))
else:
    if status.exists():
        status.rename(JOB / f"status_{int(time.time())}.json")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        [sys.executable, str(JOB / "remote_runner.py")],
        cwd=str(JOB),
        stdout=open(JOB / "remote_runner.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(json.dumps({"started": True, "pid": p.pid}))
