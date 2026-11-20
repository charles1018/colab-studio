# Colab Studio (`colab-studio`) — Repo-Level AI Guide & Skill

這是一個由 `uv` 全權管理（Python 3.12）的專案級工具，用來在使用者自己的 **Google Colab GPU** 上遠端執行開源生成模型（首發支援 **`unsloth/Qwen-Image-2.1-Turbo-GGUF`** 與官方 **Qwen 3.5 9B Prompt Enhancer**），並在每張圖完成時**自動複製一份回本地 `jobs/<job>/out/`**。

---

## 1. 環境初始化與常用指令（全部透過 `uv`）

```bash
# 第一次初始化環境並修補 google-colab-cli 0.7.4 (HTTP 411 fix)
uv sync && uv run cstudio-patch

# 離線自我測試（驗證 Graph 組裝、12 種 GGUF 量化清單與指紋）
uv run cstudio test

# 檢查 Colab 登入狀態、剩餘運算單元 (CU) 與目前開啟中的 GPU
uv run cstudio check

# 若尚未登入 Google Colab（兩步驟 OAuth）：
uv run cstudio login url          # 取得登入網址交給使用者
uv run cstudio login code <CODE>  # 貼回授權碼完成登入

# 離線預覽批次任務（不開機，檢查將下載哪些模型、哪些任務已完成或過期）
uv run cstudio plan jobs/demo_qwen21/manifest.json --gpu L4 [--quant Q4_K_M]

# 啟動 Colab GPU 生成並即時同步回本地 out/
uv run cstudio run jobs/demo_qwen21/manifest.json --gpu L4 [--quant Q4_K_M] [--session NAME]

# Session 與資源管理
uv run cstudio sessions                          # 查看開啟中的 GPU 與自動關機倒數
uv run cstudio keep --session NAME --minutes 60  # 延長保留 GPU
uv run cstudio cancel --session NAME             # 中斷雲端執行中的任務，但保留 GPU 與已載入模型
uv run cstudio stop --session NAME               # 立即關閉指定 GPU（或加 --all 關閉所有本工具開啟的機器）

# 啟動本地輕量 Web UI 工作台（預設 http://127.0.0.1:7860）
uv run cstudio ui --open
# 或：uv run cstudio-ui --port 7860

# 產生多張圖片的預覽拼圖 (Contact Sheet)
uv run cstudio sheet jobs/demo_qwen21/out/*.png
```

`cancel` 先送出指定批次的取消要求，確認遠端任務停止後才回報完成。CLI 回報等待確認時，可用 `cancel --session NAME --run-id RUN_ID` 再查同一批次。取消確認前，原有的 GPU 回收期限繼續適用。

---

## 2. 模型特性：`unsloth/Qwen-Image-2.1-Turbo-GGUF`

1. **8 步極速蒸餾 (Turbo)**：
   - 固定使用 **8 步**、**CFG 1.0**、`euler` 取樣器與官方 `ManualSigmas` (`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`)。
2. **量化等級與原生權重（預設 `"INT8"`）**：
   - 預設使用在 Colab L4/G4 上支援硬體 INT8 Tensor Core 加速、畫質與速度最佳的官方原生 **`INT8` (`qwen_image_2.1_turbo_int8_convrot.safetensors`, 7.26 GB)**。
   - 可透過 CLI `--quant <QUANT>` 或在 `manifest.json` 頂層／單一 Job 設定 `"quant"` 自由切換為官方 `BF16` (14.23G) 或 Unsloth 全部 12 種 GGUF 量化等級：
     **`INT8` (7.26G, 預設)**, `BF16` (14.23G), `Q2_K` (2.47G), `Q3_K_S` (2.72G), `Q3_K_M` (3.17G), `Q3_K_XL` (3.61G), `Q4_K_S` (3.91G), `Q4_K_M` (4.20G), `Q5_K_S` (4.50G), `Q5_K_M` (5.39G), `Q6_K` (6.27G), `Q6_K_XL` (6.72G), `Q8_0` (7.64G), `F16` (14.23G)。
3. **官方 Qwen 3.5 9B Prompt Enhancer (`refine_prompt: true`)**：
   - 當使用者只給簡短描述、或未來透過本地簡單 Web UI 操作時，在 Job 或 Manifest 設 `"refine_prompt": true`。
   - 系統會按需自動下載官方擴寫模型：
     - 文生圖 (`mode: "t2i"`)：`qwen3.5_9b_qwen_image_2.1_pe_t2i`
     - 圖生圖／改圖 (`mode: "edit"`)：`qwen3.5_9b_qwen_image_2.1_pe_i2i`（具備視覺能力，會先讀取所有 `inputs` 參考圖再生成精準改圖指令）
   - 擴寫後的完整長提示詞會自動從 ComfyUI History 提取並寫回本地 `out/results.json` 的 `jobs.<id>.refined_prompt` 欄位。
   - 可選 `"pe_variant": "int8"`（預設，9.47 GB）或 `"w4a8"`（輕量版，6.72 GB）。
4. **原生透明背景 (`transparent: true`)**：
   - Qwen-Image-2.1 的 16 通道 VAE 原生支援 4-channel RGBA 透明背景。只要在 Job 設定 `"transparent": true`，系統會自動套用官方 RGBA 提示詞包裹格式並輸出帶透明通道的 PNG。
5. **支援比例與解析度**：
   - `"aspect_ratio"`: `"1:1"`, `"4:3"`, `"3:4"`, `"3:2"`, `"2:3"`, `"16:9"`, `"9:16"`。
   - `"resolution"`: `"1K"`（約 1MP，例如 `1024x1024`、`1376x768`，速度快）或 `"2K"`（原生 4MP，例如 `2048x2048`、`2752x1536`，細節極高）。

---

## 3. `manifest.json` 範例（文生圖、透明圖、多圖改圖）

```json
{
  "model": "qwen-image-2.1-turbo",
  "quant": "Q4_K_M",
  "jobs": [
    {
      "id": "poster_t2i",
      "mode": "t2i",
      "prompt": "A neon-lit ramen shop in a rainy Tokyo alley at night, signage reading \"深夜拉麵\"",
      "aspect_ratio": "16:9",
      "resolution": "2K",
      "refine_prompt": false,
      "seed": 42
    },
    {
      "id": "cat_sticker",
      "mode": "t2i",
      "transparent": true,
      "prompt": "cyberpunk calico cat sticker",
      "aspect_ratio": "1:1",
      "resolution": "1K",
      "refine_prompt": true,
      "seed": 100
    },
    {
      "id": "edit_demo",
      "mode": "edit",
      "inputs": ["inputs/base.png", "inputs/ref_style.png"],
      "prompt": "Replace the background of <image1> with the sunset beach style from <image2>",
      "refine_prompt": true,
      "seed": 2026
    }
  ]
}
```
- **改圖 (`mode: "edit"`) 提示詞規則**：
  - 單張參考圖時直接描述如何修改該圖（或使用 `<image1>`）。
  - 多張參考圖（最多 10 張）時，在提示詞中明確以 `<image1>`, `<image2>`... 指定每張圖的角色（通常 `<image1>` 為要修改的主畫布）。
  - 預設畫布尺寸會自動跟隨第 1 張輸入圖片的比例（由 `TextEncodeQwenImage21` 的 `edit_resolution` 控制，預設 `1024`，設 `0` 則維持原圖尺寸並對齊 32 倍數）；若想強制指定輸出尺寸可加上 `"custom_size": true` 與 `"aspect_ratio"` / `"width"` / `"height"`。

---

## 4. 本地輕量 Web UI (`cstudio/webui.py` & `cstudio/static/`)

本專案內建由 **FastAPI + Uvicorn + 零建置原生 ES Module 前端** 驅動的雙欄式本地 Web UI（借鏡 Unsloth Studio 桌面版圖片生成介面）：
- **啟動方式**：`uv run cstudio ui [--host 127.0.0.1] [--port 7860] [--open]`（或 `uv run cstudio-ui`）。
- **核心功能**：
  - 頂部即時顯示 Google Colab OAuth 登入狀態（支援網頁兩步驟授權登入）、剩餘運算單元 (`CU`)、運行中 `Session` 與自動關機倒數計時，並提供 GPU (`L4`/`G4`/`A100` 等)、14 種權重規格（預設原生 `INT8`）、`+30/+60 分鐘延長保留`、`⏹️ 中斷當前任務` 與 `🛑 立即關機` 控制。
  - 左側參數設定欄支援 **文字生圖 (`T2I`)** 與 **圖片編輯／多圖參考 (`Edit`，1～10 張 `<image1>`..`<image10>` 參考圖與一鍵插入游標膠囊)**、**🔲 透明背景貼紙 (`RGBA`)**、**✨ AI 自動擴寫提示詞 (`Qwen 3.5 9B Prompt Enhancer`)**、7 種官方長寬比 (`1K`/`2K`) 與 `Edit` 四種輸出尺寸策略。
  - 右側棋盤格透明背景舞台提供即時雲端階段進度、**✨ 原始 vs 9B 擴寫後提示詞對照抽屜**、浮動玻璃工具列（一鍵 **「🎨 以此圖繼續改圖 (`<image1>`)」**、**「➕ 加入多圖參考」**、**「↺ 套用此圖設定」**）與底部本地 `jobs/` 歷史畫廊。
- **Python API 呼叫**：
  ```python
  from cstudio.runner import run_manifest, plan_batch
  from cstudio.colab_client import usage, sessions
  ```
  `run_manifest(..., progress_cb=my_callback, exit_on_error=False, should_abort=...)` 支援非 CLI 環境安全呼叫與中斷控制。

> **授權提醒**：`Qwen-Image-2.1-Turbo` 權重採用 Qwen Research License（非商業研究與評估使用）。
