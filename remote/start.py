# -*- coding: utf-8 -*-
"""Runs ON the Colab VM through `colab exec`: starts remote_runner.py in the background and returns at once."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

JOB = Path("/content/cstudio")
helpers = JOB / "job_paths.py"
if JOB.is_symlink() or helpers.is_symlink() or not helpers.is_file():
    raise RuntimeError("無法使用遠端任務路徑檢查模組")
sys.path.insert(0, str(JOB))
from job_paths import confined_path  # noqa: E402

status = confined_path(JOB, "status.json", missing_ok=True)
out_dir = confined_path(JOB, "out", directory=True, missing_ok=True)

running = subprocess.run(["pgrep", "-f", "cstudio/remote_runner.py"], capture_output=True, text=True).stdout.strip()
if running:
    # Check if the already-running process is for the same _run_id (idempotent retry after a lost exec reply)
    same_run = False
    try:
        req_run = json.loads(confined_path(JOB, "manifest.json").read_text(encoding="utf-8")).get("_run_id")
        cur_run = json.loads(status.read_text(encoding="utf-8")).get("run_id") if status.exists() else None
        same_run = bool(req_run and req_run == cur_run)
    except Exception:
        pass
    print(json.dumps({"started": same_run, "already_running": True, "same_run": same_run, "pid": running}))
else:
    worker = confined_path(JOB, "remote_runner.py")
    log_path = confined_path(JOB, "remote_runner.log", missing_ok=True)
    if status.exists():
        status.rename(confined_path(JOB, f"status_{int(time.time())}.json", missing_ok=True))
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(exist_ok=True)
    with log_path.open("w") as log:
        p = subprocess.Popen(
            [sys.executable, str(worker)],
            cwd=str(JOB),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"started": True, "pid": p.pid}))
