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
uv run cstudio login code         # 在隱藏提示中貼上授權碼，也可從 stdin 提供

# 離線預覽批次任務（不開機，檢查將下載哪些模型、哪些任務已完成或過期）
uv run cstudio plan jobs/demo_qwen21/manifest.json --gpu L4 [--quant Q4_K_M]

# 啟動 Colab GPU 生成並即時同步回本地 out/
uv run cstudio run jobs/demo_qwen21/manifest.json --gpu L4 [--quant Q4_K_M] [--session NAME]

# Session 與資源管理
uv run cstudio sessions                          # 查看開啟中的 GPU 與自動關機倒數
uv run cstudio keep --session NAME --minutes 60  # 延長保留 GPU
uv run cstudio cancel --session NAME             # 中斷雲端執行中的任務，但保留 GPU 與已載入模型
uv run cstudio recover jobs/demo_qwen21/manifest.json --run-id RUN_ID  # 補抓該批次已完成的圖
uv run cstudio stop --session NAME               # 立即關閉指定 GPU（或加 --all 關閉所有本工具開啟的機器）

# 啟動本地輕量 Web UI 工作台（預設 http://127.0.0.1:7860）
uv run cstudio ui --open
# 或：uv run cstudio-ui --port 7860

# 產生多張圖片的預覽拼圖 (Contact Sheet)
uv run cstudio sheet jobs/demo_qwen21/out/*.png
```

`cancel` 先送出指定批次的取消要求，確認遠端任務停止後才回報完成。CLI 回報等待確認時，可用 `cancel --session NAME --run-id RUN_ID` 再查同一批次。取消確認前，原有的 GPU 回收期限繼續適用。

批次中斷後，可使用當時 CLI 或工作台日誌中的批次 ID 和原工作目錄執行 `recover`。每次會補抓原 runtime 已完成且通過大小及 SHA-256 驗證的圖片。仍在執行或暫時失聯時可以重試。補抓沿用原 Colab 授權及 runtime，並受原回收期限約束。原 runtime 已消失時，仍可完成中斷前已驗證並寫入本機發布紀錄的圖片。此功能適用於已保存 `out/.runs/<run_id>/request.json` 的批次。

續跑會重驗本機圖的大小和 SHA-256。新版成果另記錄生成時的 Git 版本、Python 套件、Torch／CUDA、GPU、ComfyUI 啟動參數及模型路徑。離線規劃與續跑會比較請求指紋，並驗證保存的執行環境指紋。舊指紋相符且圖片驗證通過時，預設釘選模式可沿用舊圖，並標明其原執行環境缺少紀錄。

`plan` 與 `run` 都接受 `--no-pin-versions` 和 `--no-pin-models`。若環境未完整釘選或 ComfyUI 啟動環境無法驗證，可用 `--remake-stale` 重做，或用 `--no-fingerprint` 明確沿用已有成果。已記有大小或摘要的圖仍須通過完整性驗證。

`stop --all` 只關閉 Colab Studio 已登記的 Session。要關閉外部 Session，請明確指定 `--session NAME`。CLI 只有在所有目標均確認已關閉時才回傳成功，未確認的自有 Session 會保留 watchdog 重試紀錄。

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

CLI、工作台與 Colab worker 使用共用的 [manifest 驗證規則](schema/manifest.schema.json)。布林值與數字請用 JSON 原生型別，`width` 與 `height` 要成對指定（32～4096，生成時對齊 32）。欄位拼錯、重複或用在錯誤層級時會收到錯誤。量化、模型與解析度接受既有別名，例如 `qwen21`、`q4_k_m`、`1MP` 和 `2048`。

進階任務可指定 `cfg`、`sampler`、`sigmas` 與單張等待秒數 `timeout`。自訂 `sigmas` 使用逗號或空白分隔的十進位數字，至少兩項，依序遞減且非負。模型快取與 PE 設定可放在頂層作預設值，再由各任務覆寫。

- **改圖 (`mode: "edit"`) 提示詞規則**：
  - `inputs` 每項使用相對於 manifest 資料夾的路徑，以 `/` 分隔。可用同一資料夾的檔案與巢狀子目錄。跨 Job 沿用圖片時，可從網頁畫廊選取作品，工作台會複製到新 Job 的 `inputs/`。
  - 單張參考圖時直接描述如何修改該圖（或使用 `<image1>`）。
  - 多張參考圖（最多 10 張）時，在提示詞中明確以 `<image1>`, `<image2>`... 指定每張圖的角色（通常 `<image1>` 為要修改的主畫布）。
  - 預設畫布尺寸會自動跟隨第 1 張輸入圖片的比例（由 `TextEncodeQwenImage21` 的 `edit_resolution` 控制，預設 `1024`，設 `0` 則維持原圖尺寸並對齊 32 倍數）；若想強制指定輸出尺寸可加上 `"custom_size": true` 與 `"aspect_ratio"` / `"width"` / `"height"`。

---

## 4. 本地輕量 Web UI (`cstudio/webui.py` & `cstudio/static/`)

本專案內建由 **FastAPI + Uvicorn + 零建置原生 ES Module 前端** 驅動的雙欄式本地 Web UI（借鏡 Unsloth Studio 桌面版圖片生成介面）：
- **啟動方式**：`uv run cstudio ui [--host 127.0.0.1] [--port 7860] [--open]`（或 `uv run cstudio-ui`）。
- **部署範圍**僅限直接的本機 loopback 連線。`--host` 接受 `127.0.0.1`、`localhost`（綁定 `127.0.0.1`）與 `::1`（瀏覽器網址用 `[::1]`）。其他位址會在啟動前被拒絕。工作台信任能直接連線的本機使用者與程序。請勿透過反向代理、通道或連接埠轉送對外提供工作台。轉送器若讓請求看起來是直接的 loopback 連線，會繼承本機信任。
- 開啟首頁時，瀏覽器會取得工作台程序的 session cookie。API 與圖片需要該 cookie，程序重啟後須重新開啟首頁。控制請求須有與工作台協定、主機及埠相符的 `Origin`，並帶 `X-Colab-Studio-Request: 1`。會查詢 Colab 的狀態端點也需要此自訂標頭。工作台會拒絕代理轉送標頭。
- 請在取得 Google 登入網址的分頁貼回授權碼。登入網址有效 10 分鐘，每筆待登入流程只能交換一次。CLI 使用自己的待登入流程。提交後若交換失敗，請重新取得登入網址。成功登入會更新與 Colab CLI 共用的憑證。
- Web 改圖最多接受 10 張參考圖。上傳與畫廊沿用的圖片均限每張 35 MiB、單邊 8192 像素、總面積 16,777,216 像素。支援 PNG、JPEG、WebP、AVIF、GIF、BMP 和 TIFF，以首幀轉存 PNG。生成用 multipart 請求含表單內容共限 352 MiB，URL 編碼表單限 2 MiB。每個工作台程序同時處理一個生成請求的上傳與圖片轉檔。
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
