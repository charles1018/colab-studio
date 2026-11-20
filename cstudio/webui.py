# -*- coding: utf-8 -*-
"""FastAPI backend and StudioUIState manager for Colab Studio local Web UI."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import re
import secrets
import sys
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from ipaddress import ip_address
from pathlib import Path
from threading import Event
from typing import Any

from fastapi import APIRouter, Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from filelock import Timeout as LockTimeout
from remote.job_paths import confined_path, relative_parts
from remote.manifest import MODES, normalize_manifest, normalize_resolution

from . import colab_client as C
from . import colab_login as L
from . import runner as R
from . import web_gallery as G
from .redact import redact
from .web_inputs import (
    MAX_REFERENCES, GenerationRoute, clean_generation_staging, generation_staging, prepare_reference_images,
)
from .web_sessions import CACHE_SECONDS, STALE_SECONDS, CloudStatus, session_id, session_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC_DIR = HERE / "static"
JOBS_DIR = ROOT / "jobs"
REMOTE = ROOT / "remote"
REQUEST_DRAIN_SECONDS = 5
WORKER_DRAIN_SECONDS = C.RECOVERY_TIMEOUT
LOGGER = logging.getLogger(__name__)
if str(REMOTE) not in sys.path:
    sys.path.insert(0, str(REMOTE))
from models import qwen_image_21 as Q  # noqa: E402

STAGE_LABELS_ZH: dict[str, str] = {
    "idle": "待命中",
    "selecting_vm": "🔎 正在確認 Colab Session 目標...",
    "booting_vm": "🚀 正在 Google Colab 分配並啟動 GPU...",
    "reusing_vm": "♻️ 正在連線沿用開啟中的 Colab GPU...",
    "syncing_code": "📦 正在同步遠端引擎與任務設定...",
    "uploading_inputs": "⬆️ 正在上傳參考圖片至 Colab...",
    "starting": "▶️ 正在啟動遠端生成引擎...",
    "not started": "▶️ 正在等待遠端生成引擎回報...",
    "install": "🔧 正在安裝雲端 ComfyUI 與自訂節點環境...",
    "models": "⬇️ 正在準備模型權重...",
    "comfy": "⚙️ 正在啟動雲端 ComfyUI 伺服器...",
    "running": "🎨 正在執行雲端生成任務...",
    "syncing_outputs": "⬇️ 正在下載遠端成果並驗證大小與 SHA-256...",
    "done": "✅ 批次已完成",
    "done_with_errors": "⚠️ 有雲端任務未完成，正在收尾",
    "cancelling": "⏳ 已要求取消，等待遠端確認",
    "cancelled": "⏹️ 已確認任務取消",
    "cancel_failed": "⚠️ 無法確認遠端任務已停止",
    "stopping": "🛑 正在關閉 Colab GPU",
    "stopped": "✅ 已確認 Colab Session 關閉",
    "stopped_local": "⏹️ 已停止本地批次",
    "shutdown_pending": "⏳ Colab Session 關機待確認",
    "error": "❌ 執行發生錯誤",
    "crashed": "❌ 遠端程序異常中止",
}

PROGRESS_STEPS = {
    "selecting_vm": "boot", "booting_vm": "boot", "reusing_vm": "boot",
    "syncing_code": "boot", "uploading_inputs": "boot", "starting": "boot", "not started": "boot",
    "install": "install", "models": "models", "comfy": "generate", "running": "generate",
    "syncing_outputs": "sync", "done": "sync", "done_with_errors": "sync",
}


def get_quant_catalog() -> list[dict[str, Any]]:
    """List the 14 selectable weight formats and their pinned sizes."""
    versions = R.load_versions()
    pin_idx = {(e["repo"], e["file"]): e for e in (versions.get("models") or []) if "repo" in e and "file" in e}
    items: list[dict[str, Any]] = []

    # List native weights first, followed by the GGUF formats.
    ordered_quants = ["INT8", "BF16"] + list(Q.GGUF_QUANTS)
    descriptions = {
        "INT8": "原生 INT8 ConvRot（預設）",
        "BF16": "原生 BF16",
        "Q8_0": "Unsloth GGUF 8-bit",
        "Q6_K_XL": "Unsloth GGUF 6-bit XL",
        "Q6_K": "Unsloth GGUF 6-bit",
        "Q5_K_M": "Unsloth GGUF 5-bit Medium",
        "Q5_K_S": "Unsloth GGUF 5-bit Small",
        "Q4_K_M": "Unsloth GGUF 4-bit Medium",
        "Q4_K_S": "Unsloth GGUF 4-bit Small",
        "Q3_K_XL": "Unsloth GGUF 3-bit XL",
        "Q3_K_M": "Unsloth GGUF 3-bit Medium",
        "Q3_K_S": "Unsloth GGUF 3-bit Small",
        "Q2_K": "Unsloth GGUF 2-bit",
        "F16": "Unsloth GGUF FP16",
    }
    for q in ordered_quants:
        if q in Q.NATIVE_QUANTS:
            repo, fn, _ = Q.NATIVE_QUANTS[q]
            kind = "native"
        else:
            repo = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
            fn = Q.gguf_filename(q)
            kind = "gguf"
        entry = pin_idx.get((repo, fn)) or {}
        b = int(entry.get("bytes") or 0)
        gb = round(b / 1e9, 2) if b else None
        desc = descriptions.get(q, f"Unsloth GGUF {q}")
        label = f"{q} ({gb:.1f} GB) — {desc}" if gb else f"{q} — {desc}"
        items.append(
            {
                "id": q,
                "kind": kind,
                "file": fn.rsplit("/", 1)[-1],
                "bytes": b,
                "gb": gb,
                "description": desc,
                "label": label,
                "is_default": q == Q.DEFAULT_QUANT,
            }
        )
    return items


def read_local_keeps() -> list[dict[str, Any]]:
    """Read local ~/.config/colab-studio/keep/*.json without spawning colab CLI."""
    now = time.time()
    account = C.account_scope()
    out: list[dict[str, Any]] = []
    if not C.KEEP.is_dir():
        return out
    for p in sorted(C.KEEP.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                continue
            sess = str(d.get("session") or p.stem)
            exp = d.get("expires")
            owner = d.get("created_by", "external")
            release_pending = bool(d.get("release_pending"))
            rem_sec = max(0, int(round(exp - now))) if isinstance(exp, (int, float)) else None
            if (not account or d.get("account_scope", account) != account
                    or (owner == "owned" and not d.get("runtime_endpoint"))):
                owner, exp, rem_sec, release_pending = "unverified", None, None, False
            out.append(
                {
                    "session": sess,
                    "owner": owner,
                    "expires": exp,
                    "remaining_sec": rem_sec,
                    "busy": exp is None and owner not in ("pending", "unverified"),
                    "release_pending": release_pending,
                    "watchdog_alive": C.watchdog_alive(sess),
                }
            )
        except Exception:
            continue
    return out


def safe_subfile(job_dir: str, subfolder: str, filename: str) -> Path:
    return G.safe_subfile(JOBS_DIR, job_dir, subfolder, filename)


def scan_gallery(limit: int = 120) -> list[dict[str, Any]]:
    items, _ = G.gallery_page(JOBS_DIR, limit)
    return items


def resolve_artifact_to_path(artifact_id: str) -> Path:
    """Safely resolve a server-issued artifact_id ('<job_dir>/out/<file>' or '<job_dir>/inputs/<file>') to a local file."""
    try:
        parts = relative_parts(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="不合法的圖片識別碼") from exc
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail=f"不合法的圖片識別碼：{artifact_id!r}")
    job_dir, subfolder, filename = parts
    return safe_subfile(job_dir, subfolder, filename)


@dataclass
class GenerationAttempt:
    cancel_upload: Callable[[], None]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_open: bool = True
    preparation_cancelled: bool = False
    phase: str = "preparing"
    finished: Event = field(default_factory=Event)


class StudioUIState:
    """Thread-safe state manager for Colab Studio Web UI."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.colab_query_lock = threading.Lock()
        self.revision: int = 0
        self.view_id = uuid.uuid4().hex
        self.session_key = secrets.token_bytes(32)
        self.running: bool = False
        self.generation: GenerationAttempt | None = None
        self.closing = False
        self.provisioning: bool = False
        self.cancel_requested: bool = False
        self.abort_action: str | None = None
        self.run_id: str | None = None
        self.job_dir: str | None = None
        self.job_id: str | None = None
        self.stage: str = "idle"
        self.stage_label: str = STAGE_LABELS_ZH["idle"]
        self.progress_step: str | None = None
        self.session: str | None = None
        self.session_account: str | None = None
        self.session_endpoint: str | None = None
        self.runtime_attached = False
        self.shutdown_endpoint: str | None = None
        self.requested_gpu: str | None = None
        self.actual_gpu: str | None = None
        self.gpu_source: str | None = None
        self.quant: str = Q.DEFAULT_QUANT
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.logs: list[str] = []
        self.error: str | None = None
        self.synced_images: list[dict[str, Any]] = []
        self.last_recipe: dict[str, Any] | None = None

        self.colab_cache_account: str | None = None
        self.colab_cache = CloudStatus()
        self.cloud_generation = 0

    @contextmanager
    def reserve_generation(self, cancel_upload: Callable[[], None]):
        with self.lock:
            if self.closing:
                raise HTTPException(status_code=503, detail="工作台正在關閉")
            if self.generation or self.running:
                raise HTTPException(status_code=409, detail="已有生成請求正在接收、準備或執行")
            attempt = self.generation = GenerationAttempt(cancel_upload)
        try:
            yield attempt
        finally:
            with self.lock:
                attempt.request_open = False
                self._release_generation(attempt)

    def _release_generation(self, attempt: GenerationAttempt) -> None:
        if not attempt.request_open and attempt.phase != "running":
            self.generation = None
            attempt.finished.set()

    def _cancel_preparation(self) -> bool:
        attempt = self.generation
        if not attempt or attempt.phase != "preparing":
            return False
        if not attempt.preparation_cancelled:
            attempt.preparation_cancelled = True
            attempt.cancel_upload()
        return True

    def _request_run_cancel(self) -> None:
        self.cancel_requested = True
        self.abort_action = "cancel"
        self.stage = "cancelling"
        self.stage_label = STAGE_LABELS_ZH["cancelling"]
        self._bump()

    def begin_shutdown(self) -> Event | None:
        with self.lock:
            self.closing = True
            attempt = self.generation
            if not self._cancel_preparation() and self.running and not self.abort_action:
                self._request_run_cancel()
            return attempt.finished if attempt else None

    def abort_signal(self) -> str | None:
        with self.lock:
            return self.abort_action or ("cancel" if self.cancel_requested else None)

    def _publish_generation(
        self, attempt: GenerationAttempt, recipe: dict, account: str, endpoint: str | None,
    ) -> None:
        attempt.phase = "running"
        self.running = True
        self.provisioning = False
        self.cancel_requested = False
        self.abort_action = None
        self.run_id = None
        self.job_dir = recipe["job_dir"]
        self.job_id = recipe["job_id"]
        self.stage = "selecting_vm"
        self.stage_label = STAGE_LABELS_ZH[self.stage]
        self.progress_step = "boot"
        self.session = recipe["session"]
        self.session_account = account
        self.session_endpoint = endpoint
        self.runtime_attached = False
        self.shutdown_endpoint = None
        self.requested_gpu = recipe["gpu"]
        self.actual_gpu = None
        self.gpu_source = None
        self.quant = recipe["quant"]
        self.started_at = time.time()
        self.finished_at = None
        self.error = None
        self.synced_images = []
        self.logs = [
            f"📋 已建立本地任務：{self.job_dir} (mode={recipe['mode']}, quant={self.quant}, seed={recipe['seed']})",
            f"🎯 目標 Colab Session：{self.session or '待確認'} (GPU={self.requested_gpu})",
        ]
        self.last_recipe = recipe
        self._bump()

    def _bump(self) -> int:
        self.revision += 1
        return self.revision

    def append_log(self, msg: str) -> None:
        msg = redact(msg)
        with self.lock:
            if not self.logs or self.logs[-1] != msg:
                self.logs.append(msg)
                if len(self.logs) > 80:
                    self.logs = self.logs[-80:]
                self._bump()

    def on_runner_progress(self, st: dict[str, Any]) -> None:
        with self.lock:
            if st.get("run_id"):
                self.run_id = str(st["run_id"])
            if st.get("session"):
                self.session = str(st["session"])
            raw_stage = str(st.get("stage") or self.stage or "starting")
            endpoint = st.get("_runtime_endpoint")
            if endpoint and endpoint != self.session_endpoint:
                self.session_endpoint = endpoint
                if self.colab_cache.runtimes.get(endpoint, (None,))[0] != self.session:
                    self.invalidate_cloud()
            self.runtime_attached = st.get("_runtime_attached", self.runtime_attached)
            gpu_info = st.get("gpu_info")
            detected = gpu_info.get("name") if isinstance(gpu_info, dict) else None
            listed = st.get("_listed_hardware")
            if isinstance(detected, str) and detected.strip():
                self.actual_gpu, self.gpu_source = redact(detected.strip()), "cuda"
            elif self.actual_gpu is None and isinstance(listed, str) and listed.strip():
                self.actual_gpu, self.gpu_source = redact(listed.strip()), "colab"
            if raw_stage == "booting_vm":
                if not self.provisioning:
                    self.invalidate_cloud()
                self.provisioning = True
            elif raw_stage in ("reusing_vm", "syncing_code"):
                self.provisioning = False
            if self.abort_action != "stop":
                self.stage = raw_stage
                self.stage_label = STAGE_LABELS_ZH.get(raw_stage, f"🔄 執行中 ({raw_stage})...")
                self.progress_step = PROGRESS_STEPS.get(raw_stage, self.progress_step)

            for line in st.get("log") or []:
                s_line = redact(str(line)).strip()
                if s_line and s_line not in self.logs:
                    self.logs.append(s_line)
            if len(self.logs) > 80:
                self.logs = self.logs[-80:]

            if st.get("error") and raw_stage in ("error", "crashed", "cancelled"):
                self.error = redact(str(st["error"]))

            synced_ids = st.get("_local_synced_jobs") or []
            local_res = st.get("_local_results") or {}
            r_jobs = local_res.get("jobs") if isinstance(local_res.get("jobs"), dict) else {}
            if self.job_dir and synced_ids:
                new_synced: list[dict[str, Any]] = []
                for jid in synced_ids:
                    try:
                        png = safe_subfile(self.job_dir, "out", f"{jid}.png")
                        info = png.stat()
                    except (HTTPException, OSError):
                        continue
                    if info.st_size > 0:
                        jrec = r_jobs.get(jid) if isinstance(r_jobs.get(jid), dict) else {}
                        mtime = info.st_mtime
                        img_meta = G.inspect_image_meta(png)
                        new_synced.append(
                            {
                                "artifact_id": f"{self.job_dir}/out/{png.name}",
                                "job_dir": self.job_dir,
                                "job_id": jid,
                                "filename": png.name,
                                "url": f"/api/jobs/{self.job_dir}/out/{png.name}?t={int(mtime)}",
                                "refined_prompt": jrec.get("refined_prompt"),
                                "width": img_meta["width"] or jrec.get("width"),
                                "height": img_meta["height"] or jrec.get("height"),
                                "has_alpha": img_meta["has_alpha"],
                                "seconds": jrec.get("seconds"),
                                "sha256": jrec.get("sha256"),
                            }
                        )
                self.synced_images = new_synced

            self._bump()

    def _observe_account(self) -> str | None:
        account = C.account_scope()
        if account != self.colab_cache_account:
            self.colab_cache_account = account
            self.colab_cache = CloudStatus()
            self.cloud_generation += 1
        return account

    def invalidate_cloud(self) -> None:
        """Discard in-flight observations after an operation that can change assignments."""
        self.cloud_generation += 1
        self.colab_cache = replace(self.colab_cache, complete=False, checked_mono=None)

    def selected_endpoint(self, account: str | None, name: str, selection: str) -> str:
        """Resolve a UI selection to the exact endpoint that the browser saw."""
        with self.lock:
            if account and self._observe_account() == account:
                for endpoint, (alias, _) in self.colab_cache.runtimes.items():
                    if alias == name and selection == session_id(self.session_key, account, name, endpoint):
                        return endpoint
        raise HTTPException(status_code=409, detail="所選的 Colab Session 已變更，請重新整理並選擇")

    def _status(self, account: str | None) -> dict[str, Any]:
        now = time.time()
        cache = self.colab_cache
        fresh = cache.fresh(time.monotonic(), STALE_SECONDS)
        complete = fresh and cache.complete
        sessions = session_rows(cache, account, self.session_key, now, complete) if account else []
        same_account = bool(account and self.session_account == account)
        run_visible = same_account or self.session_account is None
        run_session = self.session if self.running and same_account else None
        run_key = (session_id(self.session_key, account, run_session, self.session_endpoint)
                   if run_session and self.session_endpoint else None)
        attached = bool(run_session and self.runtime_attached)
        elapsed = (round((now if self.running else self.finished_at) - self.started_at, 1)
                   if self.started_at is not None and (self.running or self.finished_at is not None) else 0.0)
        target = next((row for row in sessions if attached and row["id"] == run_key), None)
        gpu, gpu_source = (self.actual_gpu, self.gpu_source) if run_visible else (None, None)
        if gpu is None and target and target["presence"] == "present" and target["hardware"]:
            gpu, gpu_source = target["hardware"], "colab"
        active = target
        if active is None:
            active = next((row for row in sessions if row["presence"] != "gone"), sessions[0] if sessions else None)
        preparing = self.generation is not None and self.generation.phase == "preparing"
        return {
            "view_id": self.view_id,
            "revision": self._bump(),
            "running": self.running,
            "preparing": preparing,
            "preparation_id": self.generation.id if preparing else None,
            "can_cancel": (self.running and run_visible and not self.cancel_requested)
                          or (preparing and not self.generation.preparation_cancelled),
            "run_id": self.run_id if run_visible else None,
            "job_dir": self.job_dir,
            "job_id": self.job_id,
            "stage": self.stage,
            "stage_label": (self.stage_label if run_visible or not self.running
                            else "⏳ Colab 登入身分已變更，正在等待本機批次結束"),
            "progress_step": self.progress_step,
            "session": run_session,
            "run_session_id": run_key,
            "runtime_attached": attached,
            "requested_gpu": self.requested_gpu if run_visible else None,
            "gpu": gpu,
            "gpu_source": gpu_source,
            "quant": self.quant,
            "elapsed_sec": elapsed,
            "logs": list(self.logs[-30:]) if run_visible else [],
            "error": self.error if run_visible else None,
            "synced_images": list(self.synced_images),
            "last_recipe": self.last_recipe,
            "token_ok": bool(account),
            "cu": cache.cu if fresh else None,
            "cu_raw": cache.cu_raw if fresh else "尚未查詢",
            "cloud_status": "confirmed" if complete else "unknown",
            "checked_at": cache.checked_at,
            "confirmed_at": cache.confirmed_at,
            "sessions_failed": fresh and cache.failed,
            "cu_failed": fresh and cache.cu_failed,
            "unnamed_sessions": sum(name == "?" for name, _ in cache.runtimes.values()) if complete else 0,
            "sessions": sessions,
            "active_session": active,
        }

    def fast_status(self) -> dict[str, Any]:
        with self.lock:
            return self._status(self._observe_account())

    def refresh_colab_cache(self, force: bool = False) -> dict[str, Any]:
        with self.lock:
            account = self._observe_account()
            baseline = self.colab_cache
            if self.closing or not account or (not force and baseline.fresh(time.monotonic(), CACHE_SECONDS)):
                return self._status(account)

        # Concurrent requests for the same account/generation share the query, including force.
        with self.colab_query_lock:
            with self.lock:
                account = self._observe_account()
                cache = self.colab_cache
                if self.closing or not account or (cache is not baseline and cache.checked_mono is not None):
                    return self._status(account)
                generation = self.cloud_generation

            try:
                cu, cu_raw = C.usage(account=account)
            except (OSError, ValueError, RuntimeError, SystemExit):
                cu, cu_raw = None, "無法讀取剩餘運算單元"
            try:
                names, listing = C.sessions(check=True, account=account)
                runtimes = C.listed_runtimes(listing)
            except (OSError, RuntimeError, SystemExit):
                names, listing, runtimes = None, "", None

            with self.lock:
                current = self._observe_account()
                if self.closing or current != account or generation != self.cloud_generation:
                    return self._status(current)
                now = time.time()
                self.colab_cache = CloudStatus(
                    runtimes=runtimes if runtimes is not None else cache.runtimes,
                    complete=runtimes is not None,
                    checked_at=now,
                    checked_mono=time.monotonic(),
                    confirmed_at=now if runtimes is not None else cache.confirmed_at,
                    cu=cu if cu is not None and math.isfinite(cu) else None,
                    cu_raw=redact(cu_raw),
                    failed=runtimes is None,
                    cu_failed=cu is None or not math.isfinite(cu),
                )
                if (
                    not self.running and self.session and self.stage == "shutdown_pending"
                    and account == self.session_account and not C.keep_file(self.session).exists()
                    and C.session_gone(self.session, names, listing, self.shutdown_endpoint)
                ):
                    self.session = None
                    self.shutdown_endpoint = None
                    self.stage = "stopped"
                    self.stage_label = STAGE_LABELS_ZH["stopped"]
                return self._status(account)



STATE = StudioUIState()


@asynccontextmanager
async def workbench_lifespan(_: FastAPI):
    state = STATE
    try:
        await asyncio.to_thread(clean_generation_staging, JOBS_DIR)
    except (OSError, ValueError, LockTimeout):
        LOGGER.warning("無法清理上次未完成的工作台暫存目錄")
    with state.lock:
        state.closing = False
    try:
        yield
    finally:
        finished = state.begin_shutdown()
        if finished is not None and not await asyncio.to_thread(finished.wait, WORKER_DRAIN_SECONDS):
            LOGGER.warning("工作台收尾逾時，已登記的 Colab Session 保留既有回收紀錄")


app = FastAPI(title="Colab Studio Local Web UI", version="0.1.0", lifespan=workbench_lifespan)
app.state.reserve_generation = lambda cancel_upload: STATE.reserve_generation(cancel_upload)
LOCAL_UI_HOSTS = ("127.0.0.1", "localhost", "::1")
LOCAL_AUTHORITY = re.compile(r"(127\.0\.0\.1|localhost|\[::1\])(?::([0-9]{1,5}))?", re.IGNORECASE | re.ASCII)
UI_SESSION_TOKEN = secrets.token_urlsafe(32)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def is_loopback_peer(host: str) -> bool:
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def require_local_ui_access(request: Request, call_next):
    headers = request.headers
    if not request.client or not is_loopback_peer(request.client.host) or any(
        name in ("forwarded", "x-real-ip") or name.startswith("x-forwarded-") for name in headers
    ):
        return JSONResponse({"detail": "工作台僅接受直接的本機 loopback 連線"}, status_code=403)

    hosts = headers.getlist("host")
    authority = LOCAL_AUTHORITY.fullmatch(hosts[0]) if len(hosts) == 1 else None
    scheme = request.scope["scheme"]
    default_port = 443 if scheme == "https" else 80
    if not authority:
        return JSONResponse({"detail": "不允許此工作台 Host"}, status_code=400)
    port = int(authority[2]) if authority[2] else default_port
    if not 1 <= port <= 65535:
        return JSONResponse({"detail": "不允許此工作台 Host"}, status_code=400)

    expected_origin = f"{scheme}://{authority[1].lower()}"
    if port != default_port:
        expected_origin += f":{port}"
    origins = headers.getlist("origin")
    if origins and origins != [expected_origin]:
        return JSONResponse({"detail": "不允許跨來源存取工作台"}, status_code=403)

    path = request.scope["path"]
    root_path = request.scope.get("root_path", "")
    if root_path and path.startswith(f"{root_path}/"):
        path = path[len(root_path):]
    safe_method = request.method in ("GET", "HEAD")
    public = safe_method and (path == "/" or path.startswith("/static/"))
    cookie_name = f"cstudio_ui_{port}"
    if not public:
        fetch_sites = headers.getlist("sec-fetch-site")
        if fetch_sites and fetch_sites not in (["same-origin"], ["none"]):
            return JSONResponse({"detail": "不允許跨來源存取工作台"}, status_code=403)
        cookie = request.cookies.get(cookie_name, "")
        if not secrets.compare_digest(cookie.encode(), UI_SESSION_TOKEN.encode()):
            return JSONResponse({"detail": "請先開啟 Colab Studio 首頁"}, status_code=401)
        if not safe_method and not origins:
            return JSONResponse({"detail": "控制請求需要同源 Origin"}, status_code=403)
        # Cross-origin browser requests with this header require a CORS preflight.
        # Require it for slow status too, since that GET contacts Colab.
        if (not safe_method or path == "/api/colab/status") and headers.getlist("x-colab-studio-request") != ["1"]:
            return JSONResponse({"detail": "缺少工作台請求標頭"}, status_code=403)

    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    if not public or path == "/":
        response.headers["Cache-Control"] = "no-store"
    if path == "/" and request.method == "GET" and response.status_code == 200:
        response.set_cookie(cookie_name, UI_SESSION_TOKEN, httponly=True, samesite="strict", secure=scheme == "https")
    return response


def parse_bool_form(val: str, field: str) -> bool:
    s = val.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    raise HTTPException(status_code=400, detail=f"{field} 必須是布林值")


@app.get("/", response_class=HTMLResponse)
def index_page():
    idx = STATIC_DIR / "index.html"
    if not idx.is_file():
        return HTMLResponse("<h1>找不到 cstudio/static/index.html</h1>", status_code=500)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


@app.get("/api/meta")
def api_meta():
    """Return static model capabilities, aspect ratios, GPUs, and 14 quant options."""
    ar_list = []
    for ar, (w2, h2) in Q.ASPECT_RATIOS_2K.items():
        w1, h1 = Q.ASPECT_RATIOS_1K[ar]
        ar_list.append(
            {
                "id": ar,
                "w_1k": w1,
                "h_1k": h1,
                "w_2k": w2,
                "h_2k": h2,
            }
        )
    return {
        "gpus": list(C.GPUS),
        "default_gpu": "L4",
        "default_quant": Q.DEFAULT_QUANT,
        "quants": get_quant_catalog(),
        "aspect_ratios": ar_list,
        "pe_variants": [
            {"id": "int8", "label": "Qwen 3.5 9B INT8（預設）"},
            {"id": "w4a8", "label": "Qwen 3.5 9B W4A8"},
        ],
    }


@app.get("/api/colab/status")
def api_colab_status(force: bool = Query(False)):
    data = STATE.refresh_colab_cache(force=force)
    return data


@app.post("/api/colab/login/url")
def api_login_url():
    try:
        url, state = L.get_login_url()
        return {"ok": True, "url": url, "state": state}
    except L.LoginError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=500, detail="無法產生 Google OAuth 登入網址") from None


@app.post("/api/colab/login/code")
def api_login_code(payload: Any = Body(None)):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="登入請求格式錯誤")
    code = payload.get("code")
    state = payload.get("state")
    if not isinstance(code, str) or not code.strip():
        raise HTTPException(status_code=400, detail="請貼上 Google 授權碼")
    if not isinstance(state, str) or not state:
        raise HTTPException(status_code=400, detail=L.STATE_INVALID)
    try:
        L.submit_login_code(code, state=state)
    except L.LoginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=400, detail="Google OAuth 授權碼驗證失敗") from None
    with STATE.lock:
        STATE._observe_account()
    return {"ok": True}


def _resolve_target_session(requested: str | None) -> str:
    req = str(requested or "").strip()
    if req:
        try:
            return C.sanitize_session_name(req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    account = C.account_scope()
    with STATE.lock:
        if STATE.session and STATE.session_account in (None, account):
            try:
                return C.sanitize_session_name(STATE.session)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
    keeps = [keep for keep in read_local_keeps() if keep["owner"] not in ("pending", "unverified")]
    if len(keeps) == 1:
        return C.sanitize_session_name(keeps[0]["session"])
    names, _ = C.sessions()
    owned = [n for n in names if C.is_owned_session(n)]
    if len(owned) == 1:
        return owned[0]
    if len(names) == 1:
        return names[0]
    if not names:
        raise HTTPException(status_code=400, detail="目前沒有開啟中的 Colab Session")
    raise HTTPException(status_code=400, detail=f"有多個開啟中的 Session，請先選擇目標 Session：{', '.join(names)}")


@app.post("/api/colab/keep")
def api_colab_keep(payload: dict[str, Any]):
    try:
        minutes = float(payload.get("minutes") or 30.0)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="保留分鐘數 (minutes) 必須為數字") from e
    minutes = max(1.0, min(240.0, minutes))
    account = C.account_scope()
    session = _resolve_target_session(payload.get("session"))
    selection = payload.get("session_id")
    expected = STATE.selected_endpoint(account, session, selection) if selection is not None else None
    now = time.time()
    record = C.keep_info(session)
    cur_info = record or {}
    if cur_info.get("release_pending"):
        raise HTTPException(status_code=409, detail="此 Session 正在關機，無法延長待命")
    try:
        target = C.keep_target(session, cur_info, account)
        if expected is not None and target["extra"]["runtime_endpoint"] != expected:
            raise RuntimeError("所選的 Colab Session 已變更，請重新整理並選擇")
        cur_exp = cur_info.get("expires")
        base_time = max(now, float(cur_exp)) if isinstance(cur_exp, (int, float)) else now
        new_exp = min(now + 240.0 * 60.0, base_time + minutes * 60.0)
        C.set_keep(session, new_exp, expected=record, **target)
        C.start_watchdog(session)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    rem_sec = max(0, int(round(new_exp - now)))
    STATE.append_log(f"⏸️ 已延長 {session} 自動關機時間（目前剩餘約 {round(rem_sec / 60, 1)} 分鐘）")
    return {
        "ok": True,
        "session": session,
        "minutes": minutes,
        "remaining_sec": rem_sec,
        "local_keeps": read_local_keeps(),
    }


@app.post("/api/colab/cancel")
def api_colab_cancel(payload: dict[str, Any] | None = None):
    payload = payload or {}
    requested = str(payload.get("session") or "").strip()
    selection = payload.get("session_id")
    preparation_id = payload.get("preparation_id")
    run_id = str(payload["run_id"]).strip() if payload.get("run_id") else None
    if run_id and not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=400, detail="不合法的批次 ID")
    if requested:
        requested = _resolve_target_session(requested)
    with STATE.lock:
        if preparation_id is not None:
            attempt = STATE.generation
            if (
                selection is not None or requested or run_id or not attempt or attempt.id != preparation_id
                or not STATE._cancel_preparation()
            ):
                raise HTTPException(status_code=409, detail="此本機生成準備已變更，請重新確認")
            return JSONResponse(status_code=202, content={"ok": True, "state": "preparation_pending"})
        if selection is None and STATE.running and (not requested or requested == STATE.session):
            if STATE.session_account is not None and STATE.session_account != C.account_scope():
                raise HTTPException(status_code=409, detail="工作台的 Colab 登入身分已變更")
            if run_id and run_id != STATE.run_id:
                raise HTTPException(status_code=409, detail="工作台的執行中批次已變更")
            if STATE.abort_action == "stop":
                raise HTTPException(status_code=409, detail="此 Session 正在關機")
            STATE._request_run_cancel()
            return JSONResponse(status_code=202, content={
                "ok": True, "state": "pending", "session": STATE.session, "run_id": STATE.run_id,
            })

    target_session = _resolve_target_session(requested)
    account = C.account_scope()
    expected = STATE.selected_endpoint(account, target_session, selection) if selection is not None else None
    try:
        if expected is None:
            res = C.request_cancel(target_session, run_id)
        else:
            if C.active_endpoint(target_session, account) != expected:
                raise HTTPException(status_code=409, detail="所選的 Colab Session 已變更，請重新整理並選擇")
            res = C.request_cancel(target_session, run_id, account=account, endpoint=expected)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="未收到可確認的遠端取消結果") from exc
    state = res["state"]
    if state == "unsupported":
        raise HTTPException(status_code=409, detail="此遠端批次使用舊版取消協定")
    if state == "failed":
        raise HTTPException(status_code=409, detail="遠端未能確認指定批次取消")
    return JSONResponse(
        status_code=202 if state == "pending" else 200,
        content={"ok": True, "session": target_session, **res},
    )


@app.post("/api/colab/stop")
def api_colab_stop(payload: dict[str, Any] | None = None):
    payload = payload or {}
    stop_all = payload.get("all", False)
    requested = payload.get("session")
    if type(stop_all) is not bool or (requested is not None and not isinstance(requested, str)):
        raise HTTPException(status_code=400, detail="all 必須為布林值，session 必須為字串")
    if stop_all and requested is not None:
        raise HTTPException(status_code=400, detail="請擇一指定 all 或 session")
    if requested is not None:
        try:
            requested = C.sanitize_session_name(requested)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    account = C.account_scope()
    owned = set(C.owned_sessions()) if requested is None else set()
    with STATE.lock:
        preparation_cancelled = stop_all and STATE._cancel_preparation()
        generation_cancel_pending = (stop_all and STATE.running and STATE.stage == "selecting_vm"
                                     and STATE.session_account == account)
        if generation_cancel_pending:
            STATE._request_run_cancel()
        active = STATE.session if STATE.running else None
        if STATE.session_account is not None and STATE.session_account != account:
            if requested is not None and requested == STATE.session:
                raise HTTPException(status_code=409, detail="工作台的 Colab 登入身分已變更")
            active = None
        if active and STATE.provisioning:
            owned.add(active)
        if stop_all:
            targets = sorted(owned)
        elif requested is not None:
            targets = [requested]
        elif active in owned:
            targets = [active]
        elif len(owned) == 1:
            targets = sorted(owned)
        else:
            raise HTTPException(status_code=400, detail="請指定要關閉的 Colab Session")
        deferred = None
        if active in targets and (requested is not None or STATE.provisioning or C.is_owned_session(active)):
            STATE.cancel_requested = True
            STATE.abort_action = "stop"
            STATE.stage = "stopping"
            STATE.stage_label = STAGE_LABELS_ZH["stopping"]
            STATE._bump()
            if STATE.provisioning and (C.keep_info(active) or {}).get("created_by") != "owned":
                deferred = active

    stopped = []
    pending = []
    failed = []
    for session in targets:
        if session == deferred:
            pending.append(session)
            continue
        outcome = (C.release_runtime(session, owned_only=requested is None, account=account)
                   if account else C.ReleaseState.FAILED)
        if outcome == C.ReleaseState.CONFIRMED:
            stopped.append(session)
        elif outcome == C.ReleaseState.PENDING:
            pending.append(session)
        else:
            failed.append(session)
    with STATE.lock:
        if stopped or pending:
            STATE.invalidate_cloud()
        if not STATE.running and STATE.session in stopped and STATE.session_account in (None, account):
            STATE.session = None
            STATE.stage = "stopped"
            STATE.stage_label = STAGE_LABELS_ZH["stopped"]
        elif not STATE.running and STATE.session in pending and STATE.session_account in (None, account):
            STATE.session_account = account
            STATE.shutdown_endpoint = (C.keep_info(STATE.session) or {}).get("runtime_endpoint")
            STATE.stage = "shutdown_pending"
            STATE.stage_label = STAGE_LABELS_ZH["shutdown_pending"]
        STATE._bump()
    awaiting = bool(pending or preparation_cancelled or generation_cancel_pending)
    result = {
        "ok": not awaiting and not failed, "requested": targets, "stopped": stopped,
        "verified_gone": stopped, "pending": pending, "failed": failed,
    }
    if preparation_cancelled:
        result["preparation_cancelled"] = True
    if generation_cancel_pending:
        result["generation_cancel_pending"] = True
    if failed and not (stopped or awaiting):
        result["detail"] = f"無法確認關機：{', '.join(failed)}"
        return JSONResponse(status_code=502, content=result)
    return JSONResponse(status_code=207 if failed else 202 if awaiting else 200, content=result)


@app.get("/api/run/status")
def api_run_status():
    return STATE.fast_status()


@app.get("/api/gallery")
def api_gallery(
    limit: int = Query(120, ge=1, le=500),
    before: str | None = Query(None, max_length=250),
):
    items, next_cursor = G.gallery_page(JOBS_DIR, limit, G.parse_cursor(before))
    return {"items": items, "next_cursor": next_cursor}


@app.get("/api/gallery/thumbnail/{job_dir}/{filename}")
def api_gallery_thumbnail(job_dir: str, filename: str):
    png = safe_subfile(job_dir, "out", filename)
    return Response(G.gallery_thumbnail(png), media_type="image/png")


@app.get("/api/jobs/{job_dir}/{subfolder}/{filename}")
def api_serve_job_file(job_dir: str, subfolder: str, filename: str):
    target = safe_subfile(job_dir, subfolder, filename)
    return FileResponse(target, media_type="image/png", filename=filename)


generation = APIRouter(route_class=GenerationRoute)


@generation.post("/api/run/start")
async def api_run_start(
    request: Request,
    mode: str = Form("t2i"),
    prompt: str = Form(""),
    refine_prompt: str = Form("false"),
    pe_variant: str = Form("int8"),
    transparent: str = Form("false"),
    aspect_ratio: str = Form("1:1"),
    resolution: str = Form("1K"),
    edit_sizing: str = Form("match_1k"),
    seed: str = Form(""),
    gpu: str = Form("L4"),
    quant: str = Form("INT8"),
    session: str = Form(""),
    session_id: str = Form(""),
    idle_minutes: float = Form(15.0),
    stop_after: str = Form("false"),
    batch_name: str = Form(""),
    ref_order: str = Form("[]"),
    ref_files: list[UploadFile] = File(default=[]),
):
    """Validate inputs, create local jobs/webui_<timestamp>/ folder, and start background runner."""
    state = STATE
    attempt = request.state.generation

    if not C.token_ok():
        raise HTTPException(
            status_code=401,
            detail="尚未登入 Google Colab OAuth，請先點擊頂部狀態列的「🔑 登入 Google Colab」",
        )
    account = C.account_scope()
    if not account:
        raise HTTPException(status_code=401, detail="無法確認 Colab 登入身分，請重新登入")

    mode_clean = str(mode or "t2i").strip().lower()
    if mode_clean not in MODES:
        raise HTTPException(status_code=400, detail=f"不支援的生圖模式：{mode_clean!r}")

    prompt_clean = str(prompt or "").strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="請輸入提示詞 (Prompt)")

    gpu_clean = str(gpu or "L4").strip().upper()
    if gpu_clean not in C.GPUS:
        raise HTTPException(status_code=400, detail=f"不支援的 GPU：{gpu_clean!r}")

    quant_clean = str(quant or Q.DEFAULT_QUANT).strip().upper()
    pe_var_clean = str(pe_variant or "int8").strip().lower()

    ar_clean = str(aspect_ratio or "1:1").strip()
    if ar_clean not in Q.ASPECT_RATIOS_2K:
        raise HTTPException(status_code=400, detail=f"不支援的長寬比：{ar_clean!r}")

    try:
        res_clean = normalize_resolution(resolution)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"不支援的解析度：{exc}") from exc

    refine_bool = parse_bool_form(refine_prompt, "refine_prompt")
    transparent_bool = parse_bool_form(transparent, "transparent")
    stop_after_bool = parse_bool_form(stop_after, "stop_after")
    if not math.isfinite(idle_minutes):
        raise HTTPException(status_code=400, detail="idle_minutes 必須是有限數字")
    idle_mins = max(1.0, min(240.0, float(idle_minutes or 15.0)))

    seed_str = str(seed or "").strip()
    if seed_str:
        try:
            seed_int = int(seed_str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"種子碼 (Seed) 必須為整數：{seed_str!r}") from e
    else:
        seed_int = random.randint(1, 2_147_483_647)

    # Build safe local job directory name
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    custom_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(batch_name or "").strip()).strip("_")[:32]
    folder_name = f"webui_{ts_str}_{custom_slug}_{short_uid}" if custom_slug else f"webui_{ts_str}_{short_uid}"

    resolved_sources: list[UploadFile | Path] = []
    try:
        order_spec = json.loads(ref_order or "[]")
    except (ValueError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="ref_order 必須是參考圖順序陣列") from exc
    if not isinstance(order_spec, list):
        raise HTTPException(status_code=400, detail="ref_order 必須是參考圖順序陣列")
    if mode_clean == "t2i" and (order_spec or ref_files):
        raise HTTPException(status_code=400, detail="T2I 不接受參考圖")
    if mode_clean == "edit":
        # Map uploaded files by their client-side key or index
        upload_idx = 0
        if order_spec:
            if len(order_spec) > MAX_REFERENCES:
                raise HTTPException(status_code=400, detail=f"改圖最多接受 {MAX_REFERENCES} 張參考圖片")
            for item in order_spec:
                if not isinstance(item, dict):
                    raise HTTPException(status_code=400, detail="ref_order 包含無效的參考圖項目")
                kind = item.get("kind")
                if kind == "upload" and upload_idx < len(ref_files):
                    resolved_sources.append(ref_files[upload_idx])
                    upload_idx += 1
                elif kind == "artifact" and isinstance(item.get("artifact_id"), str) and item["artifact_id"]:
                    resolved_sources.append(resolve_artifact_to_path(item["artifact_id"]))
                else:
                    raise HTTPException(status_code=400, detail="ref_order 與提供的參考圖不符")
            if upload_idx != len(ref_files):
                raise HTTPException(status_code=400, detail="ref_order 與提供的參考圖不符")
        else:
            resolved_sources.extend(ref_files)

        if not (1 <= len(resolved_sources) <= MAX_REFERENCES):
            raise HTTPException(
                status_code=400,
                detail=f"圖片編輯（Edit）模式需要 1～{MAX_REFERENCES} 張參考圖片（目前收到 {len(resolved_sources)} 張）",
            )

    # Build job spec
    job_id = custom_slug if (custom_slug and re.fullmatch(r"[A-Za-z0-9_-]{1,60}", custom_slug)) else f"img_{ts_str}"
    job_spec: dict[str, Any] = {
        "id": job_id,
        "mode": mode_clean,
        "prompt": prompt_clean,
        "quant": quant_clean,
        "seed": seed_int,
        "refine_prompt": refine_bool,
        "pe_variant": pe_var_clean,
        "transparent": transparent_bool,
    }

    if mode_clean == "t2i":
        job_spec["aspect_ratio"] = ar_clean
        job_spec["resolution"] = res_clean
    else:
        job_spec["inputs"] = [f"inputs/ref_{index}.png" for index in range(1, len(resolved_sources) + 1)]
        sizing = str(edit_sizing or "match_1k").strip().lower()
        if sizing == "match_2k":
            job_spec["edit_resolution"] = 2048
            job_spec["custom_size"] = False
        elif sizing == "match_orig":
            job_spec["edit_resolution"] = 0
            job_spec["custom_size"] = False
        elif sizing == "custom":
            job_spec["edit_resolution"] = 2048 if res_clean == "2K" else 1024
            job_spec["custom_size"] = True
            job_spec["aspect_ratio"] = ar_clean
            job_spec["resolution"] = res_clean
        elif sizing == "match_1k":
            job_spec["edit_resolution"] = 1024
            job_spec["custom_size"] = False
        else:
            raise HTTPException(status_code=400, detail="不支援的 Edit 尺寸策略")

    manifest_data: dict[str, Any] = {
        "model": "qwen-image-2.1-turbo",
        "quant": quant_clean,
        "_gpu": gpu_clean,
        "jobs": [job_spec],
    }
    manifest_data, problems = normalize_manifest(manifest_data, None)
    if problems:
        raise HTTPException(status_code=400, detail=redact("任務設定驗證未通過：\n- " + "\n- ".join(problems)))

    # The runner selects automatic reuse from the authenticated assignment listing.
    req_sess = str(session or "").strip()
    target_session = None
    expected_endpoint = None
    if req_sess and req_sess != "__new__":
        try:
            target_session = C.sanitize_session_name(req_sess)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if isinstance(session_id, str) and session_id:
        if target_session is None:
            raise HTTPException(status_code=400, detail="請指定要沿用的 Colab Session")
        expected_endpoint = state.selected_endpoint(account, target_session, session_id)

    recipe = {
        "job_dir": folder_name,
        "job_id": job_id,
        "mode": mode_clean,
        "prompt": prompt_clean,
        "quant": quant_clean,
        "seed": str(seed_int),
        "refine_prompt": refine_bool,
        "pe_variant": pe_var_clean,
        "transparent": transparent_bool,
        "aspect_ratio": ar_clean,
        "resolution": res_clean,
        "edit_sizing": edit_sizing,
        "gpu": gpu_clean,
        "session": target_session,
    }
    launch = Event()
    committed = False

    def _worker():
        launch.wait()
        if not committed:
            return
        stage = "error"
        error = None
        clear_session = False
        log = None
        try:
            res = R.run_manifest(
                manifest_path,
                gpu=gpu_clean,
                session=target_session,
                account=account,
                auto_reuse=not req_sess,
                reuse_only=target_session is not None,
                expected_endpoint=expected_endpoint,
                cli_quant=quant_clean,
                idle_minutes=idle_mins,
                stop_after=stop_after_bool,
                progress_cb=state.on_runner_progress,
                exit_on_error=False,
                should_abort=state.abort_signal,
            )
            clear_session = res.get("shutdown_state") == C.ReleaseState.CONFIRMED
            if res.get("shutdown_state") == C.ReleaseState.PENDING:
                stage = "shutdown_pending"
            elif res.get("error"):
                error = redact(str(res["error"]))
            else:
                stage = "done"
        except BaseException as exc:
            log = f"⚠️ {redact(str(exc))}"
            if isinstance(exc, R.RunCancelled):
                stage = "cancelled"
            elif isinstance(exc, R.RunStopped) and exc.outcome != C.ReleaseState.FAILED:
                clear_session = exc.outcome == C.ReleaseState.CONFIRMED
                if clear_session:
                    stage = "stopped"
                else:
                    stage = "shutdown_pending" if exc.outcome == C.ReleaseState.PENDING else "stopped_local"
            else:
                error = redact(str(exc))
        finally:
            with state.lock:
                if clear_session:
                    state.session = None
                state.stage = stage
                state.stage_label = STAGE_LABELS_ZH[stage]
                state.error = error
                if log:
                    state.logs.append(log)
                if stage == "shutdown_pending" and state.session:
                    state.shutdown_endpoint = (
                        state.session_endpoint or (C.keep_info(state.session) or {}).get("runtime_endpoint")
                    )
                state.running = False
                state.provisioning = False
                state.finished_at = time.time()
                attempt.phase = "finished"
                state._release_generation(attempt)
                state.invalidate_cloud()
                state._bump()
            try:
                state.refresh_colab_cache(force=True)
            except Exception:
                pass

    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        job_folder = confined_path(JOBS_DIR, folder_name, directory=True, missing_ok=True)
        manifest_path = job_folder / "manifest.json"
        if job_folder.exists():
            raise FileExistsError(job_folder)
        with generation_staging(JOBS_DIR) as scratch:
            staged_job = scratch / "job"
            inputs_folder = staged_job / "inputs"
            inputs_folder.mkdir(parents=True)
            (staged_job / "out").mkdir()
            if resolved_sources:
                await prepare_reference_images(resolved_sources, inputs_folder)
            problems = R.validate_manifest(manifest_data, staged_job)
            if problems:
                raise HTTPException(status_code=400, detail=redact("任務設定驗證未通過：\n- " + "\n- ".join(problems)))
            (staged_job / "manifest.json").write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            with state.lock:
                if state.closing or attempt.preparation_cancelled:
                    raise HTTPException(status_code=409, detail="已取消本機生成準備")
                if C.account_scope() != account:
                    raise HTTPException(status_code=409, detail="Colab 登入身分已變更，請重新提交")
                if job_folder.exists():
                    raise FileExistsError(job_folder)
                # The thread can start only after the validated folder and its state are published.
                try:
                    thread = threading.Thread(target=_worker, name=f"cstudio-run-{folder_name}", daemon=True)
                    thread.start()
                except (OSError, RuntimeError) as exc:
                    LOGGER.warning("無法啟動工作台生成執行緒")
                    raise HTTPException(status_code=503, detail="無法啟動生成任務，請重試") from exc
                staged_job.rename(job_folder)
                state._publish_generation(attempt, recipe, account, expected_endpoint)
                committed = True
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="任務資料夾已經存在") from exc
    except LockTimeout as exc:
        raise HTTPException(status_code=503, detail="工作台暫存目錄暫時無法使用，請重試") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="無法使用任務資料夾") from exc
    finally:
        launch.set()

    return JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "job_dir": folder_name,
            "job_id": job_id,
            "session": target_session,
            "seed": str(seed_int),
            "status": state.fast_status(),
        },
    )


app.include_router(generation)


def launch_ui(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = False):
    import uvicorn

    host = host.lower()
    if host not in LOCAL_UI_HOSTS:
        raise SystemExit("工作台只允許 --host 127.0.0.1、localhost 或 ::1")
    bind_host = "127.0.0.1" if host == "localhost" else host
    url_host = "[::1]" if host == "::1" else host
    url = f"http://{url_host}:{port}"
    print(f"🎨 啟動 Colab Studio 本地輕量 Web UI：{url}", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    class StudioServer(uvicorn.Server):
        async def shutdown(self, sockets=None):
            STATE.begin_shutdown()
            await super().shutdown(sockets=sockets)

    config = uvicorn.Config(app, host=bind_host, port=port, log_level="info", proxy_headers=False,
                            timeout_graceful_shutdown=REQUEST_DRAIN_SECONDS)
    StudioServer(config).run()


def main():
    ap = argparse.ArgumentParser(prog="cstudio-ui", description="啟動 Colab Studio 本地輕量 Web UI")
    ap.add_argument("--host", default="127.0.0.1", help="loopback 監聽位址 (127.0.0.1、localhost 或 ::1)")
    ap.add_argument("--port", type=int, default=7860, help="監聽埠號 (預設 7860)")
    ap.add_argument("--open", action="store_true", help="啟動後自動開啟瀏覽器")
    args = ap.parse_args()
    launch_ui(host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
