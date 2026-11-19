# -*- coding: utf-8 -*-
"""CLI entrypoint for colab-studio (`uv run cstudio ...`)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import colab_client as C
from . import colab_login as L
from . import runner as R
from . import selftest as T

CANCEL_CODE = """
import json, subprocess, urllib.request
from pathlib import Path
BASE = 'http://127.0.0.1:8188'
ST = Path('/content/cstudio/status.json')
subprocess.run(['pkill', '-f', 'cstudio/remote_runner.py'], check=False)
for ep, body in [('/interrupt', b'{}'), ('/queue', b'{"clear": true}')]:
    try:
        req = urllib.request.Request(BASE + ep, body, {'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass
if ST.is_file():
    try:
        d = json.loads(ST.read_text(encoding='utf-8'))
        d.update(done=True, stage='cancelled', error='cancelled by user')
        ST.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    except Exception:
        pass
print('CANCELLED_KEEP_GPU')
"""


def cmd_check(_args):
    exe = C.colab_exe()
    print(f"🔧 colab-cli：{exe}", flush=True)
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
        print(f"⚪ 目前無開啟中的 Colab Session ({s_raw})", flush=True)


def cmd_sessions(_args):
    names, raw = C.sessions()
    if not names:
        print("⚪ 目前無開啟中的 Colab Session", flush=True)
        return
    now = time.time()
    print("🟢 目前開啟中的 Colab Sessions：", flush=True)
    for n in names:
        info = C.keep_info(n)
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
    names, _ = C.sessions()
    session = args.session
    if not session:
        owned = [n for n in names if C.is_owned_session(n)]
        if len(owned) == 1:
            session = owned[0]
        else:
            sys.exit(f"請用 --session 指定名稱（目前開啟中：{names}）")
    if session not in names:
        sys.exit(f"找不到開啟中的 session：{session}（目前開啟中：{names}）")
    mins = max(1.0, min(240.0, float(args.minutes)))
    C.set_keep(session, time.time() + mins * 60)
    C.start_watchdog(session)
    print(f"⏸️ 已將 {session} 的自動關機時間設為 {mins:g} 分鐘後", flush=True)


def cmd_stop(args):
    names, _ = C.sessions()
    if args.all:
        targets = [n for n in names if C.is_owned_session(n)]
        if not targets:
            print("⚪ 沒有由本工具開啟的 Colab Session 需要關閉", flush=True)
            return
        for t in targets:
            print(f"🛑 正在關閉 {t}...", flush=True)
            C.colab("stop", "-s", t, timeout=300, check=False)
            C.drop_keep(t)
        print("✅ 已全部關閉", flush=True)
        return

    session = args.session
    if not session:
        owned = [n for n in names if C.is_owned_session(n)]
        if len(owned) == 1:
            session = owned[0]
        else:
            sys.exit(f"請用 --session <NAME> 或 --all 指定要關閉的機器（目前開啟中：{names}）")
    print(f"🛑 正在關閉 {session}...", flush=True)
    C.colab("stop", "-s", session, timeout=300, check=False)
    C.drop_keep(session)
    print(f"✅ 已關閉 {session}", flush=True)


def cmd_cancel(args):
    names, _ = C.sessions()
    session = args.session
    if not session:
        owned = [n for n in names if C.is_owned_session(n)]
        if len(owned) == 1:
            session = owned[0]
        else:
            sys.exit(f"請用 --session <NAME> 指定要中斷任務的機器（目前開啟中：{names}）")
    if session not in names:
        sys.exit(f"找不到開啟中的 session：{session}")
    out = C.remote_exec(session, CANCEL_CODE, timeout=45, tries=3)
    C.set_keep(session, time.time() + 30 * 60)
    C.start_watchdog(session)
    if "CANCELLED_KEEP_GPU" in out:
        print(f"⏹️ 已中斷 {session} 上的雲端任務，並保留 GPU 30 分鐘（模型不需重下）", flush=True)
    else:
        print(f"⚠️ 已送出中斷指令，遠端回應：{out}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        prog="cstudio",
        description="Colab Studio: Run Qwen-Image-2.1-Turbo-GGUF and open-weight models on Google Colab GPU & sync outputs locally.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="檢查 colab-cli、OAuth 登入狀態、剩餘運算單元與開啟中的機器")

    p_login = sub.add_parser("login", help="兩步驟 Google Colab OAuth 登入 (status / url / code)")
    p_login.add_argument("action", nargs="?", default="status", choices=["status", "url", "code"])
    p_login.add_argument("auth_code", nargs="?", default=None)

    p_plan = sub.add_parser("plan", help="離線檢查 manifest、顯示將下載的模型與指紋狀態（不開機）")
    p_plan.add_argument("manifest", help="manifest.json 路徑")
    p_plan.add_argument("--gpu", default="L4", choices=C.GPUS)
    p_plan.add_argument("--quant", default=None, help="覆寫量化等級 (預設 INT8，可選 Q4_K_M, Q8_0, BF16 等)")

    p_run = sub.add_parser("run", help="在 Colab GPU 執行批次生成並即時將圖片同步回本地 out/")
    p_run.add_argument("manifest", help="manifest.json 路徑")
    p_run.add_argument("--gpu", default="L4", choices=C.GPUS)
    p_run.add_argument("--session", default=None, help="指定或沿用 Colab session 名稱")
    p_run.add_argument("--quant", default=None, help="覆寫量化等級 (預設 INT8，可選 Q4_K_M, Q8_0, BF16 等)")
    p_run.add_argument("--idle-minutes", type=float, default=15.0, help="完成後保留 GPU 待命分鐘數 (預設 15)")
    p_run.add_argument("--max-minutes", type=float, default=120.0, help="單批最長執行分鐘數上限 (預設 120)")
    p_run.add_argument("--stop-after", action="store_true", help="完成後立即關閉 Colab GPU")
    p_run.add_argument("--force", action="store_true", help="忽略指紋快取，強制重新生成所有項目")
    p_run.add_argument("--remake-stale", action="store_true", help="重新生成已變更/過期的項目（舊檔移至 out/_stale/）")
    p_run.add_argument("--no-fingerprint", action="store_true", help="只檢查檔案是否存在，不比對指紋")
    p_run.add_argument("--no-pin-versions", action="store_true", help="不釘選 ComfyUI 與節點 commit")
    p_run.add_argument("--no-pin-models", action="store_true", help="不釘選 HuggingFace 模型 revision 與大小")
    p_run.add_argument("--assign-retries", type=int, default=5)
    p_run.add_argument("--assign-wait", type=int, default=120)

    sub.add_parser("sessions", help="列出目前帳號上開啟中的 Colab GPU 與自動關機倒數")

    p_keep = sub.add_parser("keep", help="延長開啟中 Colab GPU 的保留時間")
    p_keep.add_argument("--session", default=None)
    p_keep.add_argument("--minutes", type=float, default=60.0, help="保留分鐘數 (1～240)")

    p_stop = sub.add_parser("stop", help="立即關閉 Colab GPU 釋放資源")
    p_stop.add_argument("--session", default=None)
    p_stop.add_argument("--all", action="store_true", help="關閉所有由本工具開啟的 session")

    p_cancel = sub.add_parser("cancel", help="中斷正在執行的雲端任務，但保留 Colab GPU 與已載入的模型")
    p_cancel.add_argument("--session", default=None)

    sub.add_parser("test", help="執行離線自我測試 (驗證 Graph 組裝、12 種量化清單與指紋邏輯)")

    p_sheet = sub.add_parser("sheet", help="將多張生成的圖片拼成一張預覽縮圖表 (contact sheet)")
    p_sheet.add_argument("images", nargs="+", help="圖片路徑清單")
    p_sheet.add_argument("--cols", type=int, default=3)
    p_sheet.add_argument("--width", type=int, default=512)
    p_sheet.add_argument("--out", default=None, help="輸出檔案路徑")

    p_wd = sub.add_parser("_watchdog", add_help=False)
    p_wd.add_argument("--session", required=True)

    args = ap.parse_args()

    if args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "login":
        L.handle_login(args.action, args.auth_code)
    elif args.cmd == "plan":
        R.plan_batch(args.manifest, gpu=args.gpu, cli_quant=args.quant)
    elif args.cmd == "run":
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
    elif args.cmd == "_watchdog":
        C.watchdog_loop(args.session)


if __name__ == "__main__":
    main()
