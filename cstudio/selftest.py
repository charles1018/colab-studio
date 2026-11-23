# -*- coding: utf-8 -*-
"""Offline self-test and image contact-sheet utility for colab-studio."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import runner as R

ROOT = Path(__file__).resolve().parent.parent
REMOTE = ROOT / "remote"
sys.path.insert(0, str(REMOTE))
from models import qwen_image_21 as Q  # noqa: E402


def make_image_sheet(images: list[str], cols: int = 3, thumb_width: int = 512, out_path: str | None = None) -> Path:
    paths = [Path(p) for p in images]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"找不到圖片：{p}")
    loaded = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        ratio = im.height / max(1, im.width)
        th = max(32, int(round(thumb_width * ratio)))
        im_resized = im.resize((thumb_width, th), Image.Resampling.LANCZOS)
        loaded.append((p.name, im_resized))

    cols = max(1, min(cols, len(loaded)))
    rows = (len(loaded) + cols - 1) // cols
    cell_h = max(im.height for _, im in loaded) + 28
    canvas = Image.new("RGBA", (thumb_width * cols, cell_h * rows), (24, 24, 28, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (label, im) in enumerate(loaded):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * cell_h
        canvas.paste(im, (x, y + 24), im)
        draw.rectangle([x, y, x + thumb_width, y + 22], fill=(0, 0, 0, 220))
        draw.text((x + 6, y + 4), label[:48], fill=(255, 255, 255, 255))

    dst = Path(out_path) if out_path else (paths[0].parent / "contact_sheet.png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return dst


def run_selftest() -> int:
    print("🧪 執行 colab-studio 離線自我測試...", flush=True)
    failures: list[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            failures.append(msg)
            print(f"  ❌ {msg}", flush=True)
        else:
            print(f"  ✅ {msg}", flush=True)

    # 1. Verify versions.json and all 12 GGUF quants + 2 Native quants (INT8, BF16)
    versions = R.load_versions()
    models_idx = {(e["repo"], e["file"]): e for e in (versions.get("models") or [])}
    check(len(Q.GGUF_QUANTS) == 12, f"支援全部 12 種 Unsloth GGUF 量化等級 ({', '.join(Q.GGUF_QUANTS)})")
    for q in Q.GGUF_QUANTS:
        fn = Q.gguf_filename(q)
        entry = models_idx.get(("unsloth/Qwen-Image-2.1-Turbo-GGUF", fn))
        check(
            bool(entry and entry.get("sha256") and entry.get("bytes", 0) > 1_000_000_000),
            f"versions.json 釘選 {fn} (bytes={entry.get('bytes') if entry else None})",
        )
    for q, (repo, fn, _) in Q.NATIVE_QUANTS.items():
        entry = models_idx.get((repo, fn))
        check(bool(entry and entry.get("sha256") and entry.get("bytes")), f"versions.json 釘選原生權重 {q} ({fn})")

    # 2. Verify core text encoder, VAE, and Prompt Enhancers in versions.json
    for fn in (
        f"text_encoders/{Q.CORE_TEXT_ENCODER}",
        f"text_encoders/{Q.ALT_TEXT_ENCODER}",
        f"vae/{Q.CORE_VAE}",
        f"text_encoders/{Q.PE_MODELS[('t2i', 'int8')]}",
        f"text_encoders/{Q.PE_MODELS[('t2i', 'w4a8')]}",
        f"text_encoders/{Q.PE_MODELS[('edit', 'int8')]}",
        f"text_encoders/{Q.PE_MODELS[('edit', 'w4a8')]}",
    ):
        entry = models_idx.get(("Comfy-Org/Qwen-Image-2.1", fn))
        check(bool(entry and entry.get("sha256") and entry.get("bytes")), f"versions.json 釘選 {fn}")

    # 3. Verify T2I graph (default INT8 native, no PE)
    g_t2i = Q.build_qwen21_graph(
        {"id": "t2i_basic", "mode": "t2i", "prompt": "A neon ramen shop", "aspect_ratio": "16:9", "resolution": "2K", "seed": 42}
    )
    check(g_t2i["10"]["class_type"] == "UNETLoader", "T2I 預設 INT8 使用原生 UNETLoader")
    check(
        g_t2i["10"]["inputs"]["unet_name"] == "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "預設量化等級為 INT8 (qwen_image_2.1_turbo_int8_convrot.safetensors)",
    )
    check("30" not in g_t2i and "32" not in g_t2i, "未開啟 refine_prompt 時不載入 9B Prompt Enhancer 節點")
    check(g_t2i["41"]["inputs"]["width"] == 2752 and g_t2i["41"]["inputs"]["height"] == 1536, "2K 16:9 解析度為 2752x1536")
    check(g_t2i["50"]["inputs"]["sigmas"] == Q.TURBO_SIGMAS, "套用 8 步 Turbo 專屬 ManualSigmas")
    check(g_t2i["53"]["inputs"]["cfg"] == 1.0, "預設 CFG 為 1.0")

    # 4. Verify T2I with Prompt Enhancer + Transparent RGBA + GGUF quant
    g_t2i_pe = Q.build_qwen21_graph(
        {
            "id": "t2i_pe",
            "mode": "t2i",
            "prompt": "A cyberpunk cat sticker",
            "transparent": True,
            "refine_prompt": True,
            "quant": "Q4_K_M",
        }
    )
    check(
        g_t2i_pe["10"]["class_type"] == "UnetLoaderGGUF"
        and g_t2i_pe["10"]["inputs"]["unet_name"] == "qwen-image-2.1-turbo-Q4_K_M.gguf",
        "單一 Job 可切換為 GGUF quant=Q4_K_M 並自動改用 UnetLoaderGGUF",
    )
    check(g_t2i_pe["30"]["inputs"]["clip_name"] == Q.PE_MODELS[("t2i", "int8")], "T2I refine_prompt 載入 pe_t2i 模型")
    check(g_t2i_pe["32"]["class_type"] == "TextGenerate" and g_t2i_pe["33"]["class_type"] == "PreviewAny", "串接 TextGenerate -> PreviewAny")
    check(
        g_t2i_pe["32"]["inputs"]["system_prompt"] == Q.QWEN21_TRANSPARENT_T2I_SYSTEM_PROMPT
        and set(Q.build_pe_graph(g_t2i_pe, "cutout")) == {"30", "32", "33"},
        "透明 T2I 先擴寫主體，再用 RGBA 包裝出圖",
    )
    cutout = Q.build_qwen21_graph({"id": "cutout", "prompt": "A red mug", "transparent": True})
    check(cutout["40"]["inputs"]["prompt"] == Q.wrap_transparent_prompt("A red mug"),
          "TextEncodeQwenImage21 接收 RGBA 包裝後的文字")

    # 5. Verify Image Edit with multi-reference + Prompt Enhancer
    g_edit = Q.build_qwen21_graph(
        {
            "id": "edit_pe",
            "mode": "edit",
            "inputs": ["a.png", "b.png"],
            "prompt": "Put the character from <image2> into <image1>",
            "refine_prompt": True,
            "pe_variant": "w4a8",
        }
    )
    check("20" in g_edit and "21" in g_edit, "Edit 模式自動建立對應數量的 LoadImage 節點")
    check(g_edit["30"]["inputs"]["clip_name"] == Q.PE_MODELS[("edit", "w4a8")], "Edit refine_prompt 支援 pe_i2i w4a8 模型")
    check(g_edit["31"]["class_type"] == "BatchImagesNode", "Edit refine_prompt 使用 BatchImagesNode 將多張參考圖送入 TextGenerate")
    check(g_edit["54"]["inputs"]["latent_image"] == ["40", 2], "Edit 預設使用 TextEncodeQwenImage21 slot 2 (第一張參考圖尺寸) 作為 latent")

    # 6. Verify required_models_for_batch
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        img_p = tmp_p / "ref.png"
        Image.new("RGB", (64, 64), "blue").save(img_p)
        manifest = {
            "model": "qwen-image-2.1-turbo",
            "jobs": [
                {"id": "j1", "mode": "t2i", "prompt": "hello"},
                {"id": "j2", "mode": "edit", "inputs": ["ref.png"], "prompt": "edit <image1>", "refine_prompt": True},
            ],
        }
        probs = R.validate_manifest(manifest, tmp_p)
        check(not probs, f"manifest 離線驗證通過 ({probs})")
        req = Q.required_models_for_batch(manifest)
        req_files = [x[1] for x in req]
        check(
            "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors" in req_files
            and f"text_encoders/{Q.PE_MODELS[('edit', 'int8')]}" in req_files
            and f"text_encoders/{Q.PE_MODELS[('t2i', 'int8')]}" not in req_files,
            "按需下載模型清單正確（僅下載批次實際用到的 INT8/GGUF 與對應的 PE 模型）",
        )
        fp1 = R.compute_job_fingerprint(manifest["jobs"][1], manifest, tmp_p, versions=versions)
        fp2 = R.compute_job_fingerprint(manifest["jobs"][1], manifest, tmp_p, cli_quant="Q4_K_M", versions=versions)
        changed_versions = dict(versions, comfyui=dict(versions["comfyui"], commit="fixture-commit"))
        fp3 = R.compute_job_fingerprint(manifest["jobs"][1], manifest, tmp_p, versions=changed_versions)
        Image.new("RGB", (64, 64), "red").save(img_p)
        fp4 = R.compute_job_fingerprint(manifest["jobs"][1], manifest, tmp_p, versions=versions)
        check(fp1 not in (fp2, fp3, fp4), "Job 指紋會隨量化等級、ComfyUI 版本或參考圖內容變更")

    # 7. Verify Local Web UI (FastAPI routes, static assets, exit_on_error=False, and gallery scan)
    from . import colab_client as C
    from . import web_gallery as G
    from . import webui as W

    check((W.STATIC_DIR / "index.html").is_file(), "Web UI 靜態檔案 index.html 存在")
    check((W.STATIC_DIR / "style.css").is_file(), "Web UI 靜態檔案 style.css 存在")
    check((W.STATIC_DIR / "app.js").is_file(), "Web UI 靜態檔案 app.js 存在")

    meta = W.api_meta()
    check(len(meta["quants"]) == 14 and meta["default_quant"] == "INT8", "Web UI /api/meta 提供完整 14 種權重規格清單（預設 INT8）")
    check(len(meta["aspect_ratios"]) == 7, "Web UI /api/meta 提供 7 種官方 1K/2K 長寬比設定")

    # Verify sanitize_session_name rejects path traversal
    rejected_traversal = False
    try:
        C.sanitize_session_name("../../etc/passwd")
    except ValueError:
        rejected_traversal = True
    check(rejected_traversal, "sanitize_session_name 成功攔截路徑穿越攻擊字串")

    # Verify exit_on_error=False raises RuntimeError instead of SystemExit
    raised_runtime_err = False
    try:
        R.load_manifest("/nonexistent/path/manifest.json", exit_on_error=False)
    except RuntimeError:
        raised_runtime_err = True
    check(raised_runtime_err, "load_manifest(exit_on_error=False) 拋出 RuntimeError 而非殺死 Web Server 執行緒")

    # Verify parse_sessions ignores '[colab] Pruned 1 stale local session(s).'
    pruned_parsed = C.parse_sessions("[colab] Pruned 1 stale local session(s).\n[colab] No active sessions found on server.")
    check(pruned_parsed == [], "parse_sessions 正確忽略 [colab] Pruned / No active sessions 系統訊息")

    with tempfile.TemporaryDirectory() as tmp:
        jobs = Path(tmp)
        out = jobs / "fixture" / "out"
        out.mkdir(parents=True)
        with Image.new("RGBA", (8, 8), (0, 0, 255, 127)) as image:
            image.save(out / "sample.png")
            image.save(out / "contact_sheet.png")
        gallery, _ = G.gallery_page(jobs, 50)
        check(
            [item["artifact_id"] for item in gallery] == ["fixture/out/sample.png"],
            "畫廊掃描可讀取暫存成果並排除拼圖表",
        )

    if failures:
        print(f"\n❌ 自我測試失敗（共 {len(failures)} 項錯誤）", flush=True)
        return 1
    print("\n🎉 所有離線自我測試皆通過！", flush=True)
    return 0
