# -*- coding: utf-8 -*-
"""CLI entrypoint for colab-studio (`uv run cstudio ...`)."""
from __future__ import annotations

import argparse
import sys
import time

from . import colab_client as C
from . import colab_login as L
from . import runner as R
from . import selftest as T
from .patch_colab_cli import patch_status
from .redact import redact

def cmd_check(_args):
    status, _ = patch_status()
    if "error" in status:
        print(f"🔧 colab-cli：{status['error']}", flush=True)
    else:
        print(f"🔧 colab-cli {status['version']}：{status['package']}", flush=True)
        states = ", ".join(f"{name}={state}" for name, state in status["files"].items())
        print(f"   補丁狀態：{states}（詳見 `cstudio-patch --status`）", flush=True)
    if not C.token_ok():
        print("❌ 尚未登入 Google Colab。請執行：`uv run cstudio login url`", flush=True)
        sys.exit(1)
    print("✅ Google OAuth token 已就緒", flush=True)
    cu, raw = C.usage()
    if cu is not None:
        print(f"💳 剩餘運算單元：{cu} CU", flush=True)
    else:
        print(f"💳 運算單元狀態：{raw}", flush=True)
    names, s_raw = C.sessions()
    if names:
        print(f"🟢 目前開啟中的 Colab Sessions：{', '.join(names)}", flush=True)
    else:
        print(s_raw, flush=True)


def cmd_sessions(_args):
    account = C.account_scope()
    names, raw = C.sessions(account=account)
    if not names:
        print(raw, flush=True)
        return
    now = time.time()
    assignments = C.listed_assignments(raw) or {}
    print("🟢 目前開啟中的 Colab Sessions：", flush=True)
    for n in names:
        info = C.keep_info(n)
        if info and (info.get("account_scope", account) != account
                     or assignments.get(info.get("runtime_endpoint")) != n):
            info = None
        exp = info.get("expires") if info else None
        owner = (info or {}).get("created_by", "unknown")
        if exp is None and info:
            rem = "執行任務中"
        elif isinstance(exp, (int, float)):
            left_m = max(0.0, (exp - now) / 60.0)
            rem = f"剩餘約 {left_m:.1f} 分鐘自動關機"
        else:
            rem = "無自動關機計時器"
        print(f"  - [{n}] ({owner})：{rem}", flush=True)


def cmd_keep(args):
    account = C.account_scope()
    names, raw = C.sessions(account=account)
    session = args.session
    if not session:
        owned = [n for n in names if C.is_owned_session(n, account=account, assignments=C.listed_assignments(raw))]
        if len(owned) == 1:
            session = owned[0]
        else:
            sys.exit(f"請用 --session 指定名稱（目前開啟中：{names}）")
    if session not in names:
        sys.exit(f"找不到開啟中的 session：{session}（目前開啟中：{names}）")
    record = C.keep_info(session)
    info = record or {}
    if info.get("release_pending"):
        sys.exit(f"{session} 正在關機，無法延長待命")
    try:
        target = C.keep_target(session, info, account)
        mins = max(1.0, min(240.0, float(args.minutes)))
        C.set_keep(session, time.time() + mins * 60, expected=record, **target)
        C.start_watchdog(session)
    except (OSError, RuntimeError) as exc:
        sys.exit(str(exc))
    print(f"⏸️ 已將 {session} 的自動關機時間設為 {mins:g} 分鐘後", flush=True)


def cmd_stop(args):
    account = C.account_scope()
    if args.session:
        try:
            targets = [C.sanitize_session_name(args.session)]
        except ValueError as exc:
            sys.exit(str(exc))
    else:
        targets = C.owned_sessions()
        if not args.all and len(targets) != 1:
            sys.exit("請用 --session <NAME> 或 --all 指定要關閉的機器")
    if not targets:
        print("⚪ 目前沒有已登記的 Colab Studio Session 需要關閉", flush=True)
        return
    if not account:
        sys.exit("無法確認 Colab 登入身分")

    all_confirmed = True
    for session in targets:
        print(f"🛑 正在關閉 {session}...", flush=True)
        outcome = C.release_runtime(session, owned_only=args.session is None, account=account)
        if outcome == C.ReleaseState.CONFIRMED:
            print(f"✅ 已確認 {session} 關閉", flush=True)
        else:
            all_confirmed = False
            if outcome == C.ReleaseState.PENDING:
                print(f"⏳ {session} 尚待確認，watchdog 將重試", flush=True)
            else:
                print(f"⚠️ 無法確認 {session} 關閉，請檢查遠端狀態", flush=True)
    if not all_confirmed:
        sys.exit(1)


def cmd_cancel(args):
    account = C.account_scope()
    if not account:
        sys.exit("無法確認 Colab 登入身分")
    names, raw = C.sessions(account=account)
    session = args.session
    if not session:
        owned = [n for n in names if C.is_owned_session(n, account=account, assignments=C.listed_assignments(raw))]
        if len(owned) == 1:
            session = owned[0]
        else:
            sys.exit(f"請用 --session <NAME> 指定要中斷任務的機器（目前開啟中：{names}）")
    if session not in names:
        sys.exit(f"找不到開啟中的 session：{session}")
    try:
        res = C.request_cancel(session, args.run_id, account=account)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(f"無法確認 {session} 的取消狀態：{exc}")
    run_id = res.get("run_id")
    state = res["state"]
    if state == "confirmed":
        print(f"⏹️ 已確認 {session} 的批次 {run_id} 取消", flush=True)
    elif state == "pending":
        print(f"⏳ {session} 的批次 {run_id} 已收到取消要求，等待遠端確認。", flush=True)
    elif state in ("idle", "finished"):
        print(f"{session} 上沒有符合條件的執行中批次。", flush=True)
    else:
        sys.exit(f"無法確認 {session} 的批次 {run_id} 已停止（{state}）")


def cmd_recover(args):
    try:
        report = R.recover_outputs(args.manifest, args.run_id)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(redact(str(exc)))
    print(f"批次 {args.run_id} 的成果補抓", flush=True)
    for key, label in (
        ("synced", "已補抓"), ("already_synced", "已在本機並通過驗證"), ("pending", "待確認或待完成"),
        ("unavailable", "遠端已無法補抓"), ("superseded", "本機已有其他批次的成果"),
    ):
        if report[key]:
            print(f"  {label}：{', '.join(report[key])}", flush=True)
    if report.get("remote_error"):
        print(report["remote_error"], flush=True)
    for jid, error in report["errors"].items():
        print(f"  {jid}：{error}", flush=True)
    if report["pending"] or report["unavailable"] or report["errors"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        prog="cstudio",
        description="Colab Studio: Run Qwen-Image-2.1-Turbo-GGUF and open-weight models on Google Colab GPU & sync outputs locally.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="檢查 colab-cli、OAuth 登入狀態、剩餘運算單元與開啟中的機器")

    p_login = sub.add_parser("login", help="兩步驟 Google Colab OAuth 登入 (status / url / code)")
    p_login.add_argument("action", nargs="?", default="status", choices=["status", "url", "code"])
    p_login.add_argument("auth_code", nargs="?", default=None, help="舊版位置參數，省略時由隱藏提示或 stdin 讀取")

    p_plan = sub.add_parser("plan", help="離線檢查 manifest、顯示將下載的模型與指紋狀態（不開機）")
    p_plan.add_argument("manifest", help="manifest.json 路徑")
    p_plan.add_argument("--gpu", default="L4", choices=C.GPUS)
    p_plan.add_argument("--quant", default=None, help="覆寫量化等級 (預設 INT8，可選 Q4_K_M, Q8_0, BF16 等)")

    p_run = sub.add_parser("run", help="在 Colab GPU 生成，驗證後逐項同步圖片至本地 out/")
    p_run.add_argument("manifest", help="manifest.json 路徑")
    p_run.add_argument("--gpu", default="L4", choices=C.GPUS)
    p_run.add_argument("--session", default=None, help="指定或沿用 Colab session 名稱")
    p_run.add_argument("--quant", default=None, help="覆寫量化等級 (預設 INT8，可選 Q4_K_M, Q8_0, BF16 等)")
    p_run.add_argument("--idle-minutes", type=float, default=15.0, help="完成後保留 GPU 待命分鐘數 (預設 15)")
    p_run.add_argument("--max-minutes", type=float, default=120.0, help="單批最長執行分鐘數上限 (預設 120)")
    p_run.add_argument("--stop-after", action="store_true", help="批次收尾時請求關閉 Colab Session 並查驗結果")
    p_run.add_argument("--force", action="store_true", help="忽略指紋快取，強制重新生成所有項目")
    p_run.add_argument("--remake-stale", action="store_true", help="重新生成已變更/過期的項目（舊檔移至 out/_stale/）")
    p_run.add_argument("--no-fingerprint", action="store_true", help="略過生成指紋比對，仍驗證記有摘要的本機圖")
    for parser in (p_plan, p_run):
        parser.add_argument("--no-pin-versions", action="store_true", help="不釘選 ComfyUI 與節點 commit")
        parser.add_argument("--no-pin-models", action="store_true", help="略過 Hugging Face 模型的 revision、大小與 SHA-256 釘選")
    p_run.add_argument("--assign-retries", type=int, default=5)
    p_run.add_argument("--assign-wait", type=int, default=120)

    p_recover = sub.add_parser("recover", help="從原 Colab runtime 補抓指定批次已完成的成果")
    p_recover.add_argument("manifest", help="原工作目錄的 manifest.json 路徑")
    p_recover.add_argument("--run-id", required=True, help="執行時顯示的批次 ID")

    sub.add_parser("sessions", help="列出目前帳號上開啟中的 Colab GPU 與自動關機倒數")

    p_keep = sub.add_parser("keep", help="延長開啟中 Colab GPU 的保留時間")
    p_keep.add_argument("--session", default=None)
    p_keep.add_argument("--minutes", type=float, default=60.0, help="保留分鐘數 (1～240)")

    p_stop = sub.add_parser("stop", help="請求關閉 Colab Session 並查驗結果")
    stop_target = p_stop.add_mutually_exclusive_group()
    stop_target.add_argument("--session", default=None)
    stop_target.add_argument("--all", action="store_true", help="關閉所有由本工具開啟的 session")

    p_cancel = sub.add_parser("cancel", help="中斷正在執行的雲端任務，但保留 Colab GPU 與已載入的模型")
    p_cancel.add_argument("--session", default=None)
    p_cancel.add_argument("--run-id", default=None, help="限制取消目標為指定批次")

    sub.add_parser("test", help="執行離線自我測試 (驗證 Graph 組裝、12 種量化清單與指紋邏輯)")

    p_sheet = sub.add_parser("sheet", help="將多張生成的圖片拼成一張預覽縮圖表 (contact sheet)")
    p_sheet.add_argument("images", nargs="+", help="圖片路徑清單")
    p_sheet.add_argument("--cols", type=int, default=3)
    p_sheet.add_argument("--width", type=int, default=512)
    p_sheet.add_argument("--out", default=None, help="輸出檔案路徑")

    p_ui = sub.add_parser("ui", help="啟動本地輕量 Web UI 介面 (借鏡 Unsloth Studio 雙欄設計)")
    p_ui.add_argument("--host", default="127.0.0.1", help="loopback 監聽位址 (127.0.0.1、localhost 或 ::1)")
    p_ui.add_argument("--port", type=int, default=7860, help="監聽埠號 (預設 7860)")
    p_ui.add_argument("--open", action="store_true", help="啟動後自動開啟瀏覽器")

    p_wd = sub.add_parser("_watchdog", add_help=False)
    p_wd.add_argument("--session", required=True)
    p_wd.add_argument("--ticket")

    args = ap.parse_args()

    if args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "login":
        L.handle_login(args.action, args.auth_code)
    elif args.cmd == "plan":
        R.plan_batch(args.manifest, gpu=args.gpu, cli_quant=args.quant,
                     pin_versions=not args.no_pin_versions, pin_models=not args.no_pin_models)
    elif args.cmd == "run":
        try:
            R.run_manifest(
                args.manifest,
                gpu=args.gpu,
                session=args.session,
                cli_quant=args.quant,
                idle_minutes=args.idle_minutes,
                max_minutes=args.max_minutes,
                stop_after=args.stop_after,
                force=args.force,
                remake_stale=args.remake_stale,
                use_fingerprint=not args.no_fingerprint,
                pin_versions=not args.no_pin_versions,
                pin_models=not args.no_pin_models,
                assign_retries=args.assign_retries,
                assign_wait=args.assign_wait,
            )
        except (C.AllocationServiceUnavailable, R.RunCancelled) as exc:
            raise SystemExit(str(exc)) from None
    elif args.cmd == "recover":
        cmd_recover(args)
    elif args.cmd == "sessions":
        cmd_sessions(args)
    elif args.cmd == "keep":
        cmd_keep(args)
    elif args.cmd == "stop":
        cmd_stop(args)
    elif args.cmd == "cancel":
        cmd_cancel(args)
    elif args.cmd == "test":
        sys.exit(T.run_selftest())
    elif args.cmd == "sheet":
        dst = T.make_image_sheet(args.images, cols=args.cols, thumb_width=args.width, out_path=args.out)
        print(f"🖼️ 已產生預覽拼圖：{dst}", flush=True)
    elif args.cmd == "ui":
        from . import webui as W

        W.launch_ui(host=args.host, port=args.port, open_browser=args.open)
    elif args.cmd == "_watchdog":
        C.watchdog_loop(args.session, args.ticket)


if __name__ == "__main__":
    main()
