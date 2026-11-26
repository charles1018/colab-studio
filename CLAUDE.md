# Colab Studio (`colab-studio`) — Repo-Level AI Guide & Skill

Colab Studio 使用 Python 3.12 與 `uv`，在使用者自己的 Google Colab GPU 執行 Qwen-Image-2.1 Turbo，提供原生 INT8／BF16、Unsloth GGUF 和 Qwen 3.5 9B Prompt Enhancer。成功完成的圖片會逐項下載，通過大小及 SHA-256 驗證後存入 manifest 資料夾的 `out/`。

## 維護與驗收原則

- 開工前先寫清楚這批要驗證的行為、完成所需的證據，以及未通過時最多重試幾次。完成就收尾，次數用完就停並記錄結論。
- 模型畫質、硬體容量及平台相容性等限制，依證據寫入驗證範圍或已知限制。確認 Studio 行為符合預期後，以限制結案。
- 先用 `cstudio plan`、`run`、`recover`、`stop`、工作台或既有測試驗證。若有可重現的產品缺陷，補能覆蓋該缺陷的精簡回歸測試，避免另建整套測試工具或控制器。
- 連續兩批都沒有 repo 變更時，與使用者回顧這些批次回答了什麼、剩下的問題值不值得再做。檢討完成前，暫停核准及執行下一批。
- 延後項目見 [ROADMAP.md](ROADMAP.md)，只在觸發時機出現時處理。

---

## 1. 環境初始化與常用指令（全部透過 `uv`）

```bash
# 第一次初始化環境並修補 google-colab-cli 0.7.4 (HTTP 411 fix)
uv sync --locked
uv run --locked --offline --no-sync cstudio-patch

# 唯讀查詢本機補丁狀態、執行離線自我測試
uv run --locked --offline --no-sync cstudio-patch --status
uv run --locked --offline --no-sync cstudio test

# 檢查 Colab 登入狀態、剩餘運算單元 (CU) 與目前開啟中的 GPU
uv run cstudio check

# 若尚未登入 Google Colab（兩步驟 OAuth）：
uv run cstudio login url          # 取得登入網址交給使用者
uv run cstudio login code         # 在隱藏提示中貼上授權碼，也可從 stdin 提供

# 離線預覽批次任務（不開機，檢查將下載哪些模型、哪些任務已完成或過期）
uv run --locked --offline --no-sync cstudio plan jobs/demo_qwen21/manifest.json --gpu L4 [--quant Q4_K_M]

# 啟動 Colab GPU 生成，逐項驗證並同步回本地 out/
uv run cstudio run jobs/demo_qwen21/manifest.json --gpu L4 [--quant Q4_K_M] [--session NAME]

# Session 與資源管理
uv run cstudio sessions                          # 查看開啟中的 GPU 與自動關機倒數
uv run cstudio keep --session NAME --minutes 60  # 延長保留 GPU
uv run cstudio cancel --session NAME             # 中斷雲端執行中的任務，但保留 GPU 與已載入模型
uv run cstudio recover jobs/demo_qwen21/manifest.json --run-id RUN_ID  # 補抓該批次已完成的圖
uv run cstudio stop --session NAME               # 請求關閉指定 Session 並查驗結果，--all 只處理已登記的自有 Session

# 啟動本地輕量 Web UI 工作台（預設 http://127.0.0.1:7860）
uv run cstudio ui --open
# 或：uv run cstudio-ui --port 7860

# 產生多張圖片的預覽拼圖 (Contact Sheet)
uv run cstudio sheet jobs/demo_qwen21/out/*.png
```

`cstudio-patch --status` 回傳逐檔 JSON 狀態。退出碼 `0` 表示已補齊，`1` 表示待補（含 `legacy`），`2` 表示不相容。重建虛擬環境後可重新查詢。Studio 的帳號綁定子程序會在程序內停用上游 CLI 的更新檢查，共用使用者設定可獨立管理。

Linux 已實測 Colab OAuth 與 CLI，也以本機瀏覽器驗證工作台。已知平台限制為 macOS 端到端流程、Windows 相容性及 Windows ACL 未驗證。釘選的 Colab CLI 上游列出 Linux 與 macOS。 [google-colab-cli 0.7.4 平台說明](https://pypi.org/project/google-colab-cli/0.7.4/)

`cancel` 先送出指定批次的取消要求，確認遠端任務停止後才回報完成。CLI 回報等待確認時，可用 `cancel --session NAME --run-id RUN_ID` 再查同一批次。取消確認前，原有的 GPU 回收期限繼續適用。

批次中斷後，可使用當時 CLI 或工作台日誌中的批次 ID 和原工作目錄執行 `recover`。每次會補抓原 runtime 已完成且通過大小及 SHA-256 驗證的圖片。仍在執行或暫時失聯時可以重試。補抓沿用原 Colab 授權及 runtime，並受原回收期限約束。原 runtime 已消失時，仍可完成中斷前已驗證並寫入本機發布紀錄的圖片。此功能適用於已保存 `out/.runs/<run_id>/request.json` 的批次。

續跑會重驗本機圖的大小和 SHA-256。新版成果另記錄生成時的 Git 版本、Python 套件、Torch／CUDA、GPU、ComfyUI 啟動參數及模型路徑。離線規劃與續跑會比較請求指紋，並驗證保存的執行環境指紋。舊指紋相符且圖片驗證通過時，預設釘選模式可沿用舊圖，並標明其原執行環境缺少紀錄。

`plan` 與 `run` 預設釘選 ComfyUI、自訂節點及模型權重。兩個命令都接受 `--no-pin-models`（只停用權重釘選）及 `--no-pin-versions`（同時停用版本及權重釘選）。`remote/versions.json` 的 `CSTUDIO_PIN_VERSIONS=0` 是舊備註，現行入口以 CLI 旗標為準。

若環境未完整釘選或 ComfyUI 啟動環境無法驗證，`run` 可用 `--remake-stale` 重做，或用 `--no-fingerprint` 明確沿用已有成果。已記有大小或摘要的圖仍須通過完整性驗證。

`stop --all` 只關閉 Colab Studio 已登記的 Session。要關閉外部 Session，請明確指定 `--session NAME`。CLI 只有在所有目標均確認已關閉時才回傳成功，未確認的自有 Session 會保留 watchdog 重試紀錄。

---

## 2. 模型設定與驗證範圍

**Turbo 預設。** Studio 預設使用 8 步、CFG 1.0、`euler` 與 `ManualSigmas` 排程 `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`。每個 Job 可透過 `cfg`、`sampler` 與 `sigmas` 覆寫。

**權重規格。** 原生 INT8 ConvRot（Studio 預設，7.26 GB）與 BF16（14.23 GB）使用 `UNETLoader`。Unsloth 的 12 種 GGUF 規格使用 `UnetLoaderGGUF`，包含 `Q2_K`、`Q3_K_S`、`Q3_K_M`、`Q3_K_XL`、`Q4_K_S`、`Q4_K_M`、`Q5_K_S`、`Q5_K_M`、`Q6_K`、`Q6_K_XL`、`Q8_0` 和 `F16`。選取順序為 CLI `--quant`、Job 的 `quant`、manifest 頂層 `quant`、Studio 預設。檔案大小見版本表，也可用 `plan` 查看當批需要的權重檔案和合計大小。 [版本表](remote/versions.json)

**Qwen 3.5 9B Prompt Enhancer。** 啟用 `refine_prompt: true` 後，T2I 使用 `pe_t2i`，Edit 使用 `pe_i2i` 並依序傳入 1～10 張參考圖的白底 RGB 合成。Edit 的生成編碼使用參考圖的 RGBA。擴寫模型可選 `pe_variant: "int8"`（Studio 預設，約 9.47 GB）或 `"w4a8"`（約 6.72 GB）。ComfyUI History 回傳的擴寫文字會存到 `out/results.json` 的 `jobs.<id>.refined_prompt`。 [Comfy-Org 模型檔案](https://huggingface.co/Comfy-Org/Qwen-Image-2.1#prompt-enhancer)

**RGBA。** `transparent: true` 會為生成提示詞套用透明背景格式。釘選的 ComfyUI 將 Qwen-Image-2.1 VAE 識別為 64 個 latent 通道與 16 倍空間縮放，解碼可輸出 RGBA，PNG 保存解碼結果。實際 alpha 像素由模型生成。 [釘選版 ComfyUI VAE](https://github.com/Comfy-Org/ComfyUI/blob/d5d59c4011312df6b3a4241fb48c045289ad9441/comfy/sd.py#L794-L807)

**尺寸預設。** 1K（約 1 MP）與 2K（約 4 MP）各有對齊 32 的尺寸表。T2I 和 Edit 自訂尺寸選項使用下表。

| 長寬比 | 1K | 2K |
| --- | --- | --- |
| 1:1 | 1024×1024 | 2048×2048 |
| 4:3 | 1216×896 | 2400×1792 |
| 3:4 | 896×1216 | 1792×2400 |
| 3:2 | 1280×832 | 2528×1696 |
| 2:3 | 832×1280 | 1696×2528 |
| 16:9 | 1376×768 | 2752×1536 |
| 9:16 | 768×1376 | 1536×2752 |

Linux 離線測試驗證 manifest、生成圖組裝、權重選取、成果同步及 Session 契約的替身情境。CLI 與工作台可請求 L4（Studio 預設）、G4、A100、H100、T4，實際可用規格由 Colab 決定。

Colab L4 搭配 Q2_K 已實測 T2I、RGBA（含透明像素）、提示詞擴寫（PE），以及 1～10 張 Edit 參考圖的有序上傳、SHA-256 驗證、ComfyUI history 和成果同步。參考圖較多時，尤其是 9～10 張，構圖、編號和挖孔位置受模型及量化限制，Studio 不保證精確重現。

L4（22.03 GiB）上，關閉 PE 時，Studio 預設的原生 INT8 UNet 與 Q4_K_M 均通過 1024×1024 的 T2I 及雙圖 Edit。INT8 另通過 `edit_resolution: 0` 的 Edit 首圖尺寸（768×512）。L4 上的預設 INT8 組合先前以 `--gpu-only` 實測通過 2048×2048 的 2K T2I。現行會使用 `normal-vram`，此模式下的 2K 尚未實測。已實測的成果均通過同步、SHA-256、ComfyUI history 和基本目視檢查。

BF16 搭配預設 INT8 文字編碼器及 VAE 時，權重合計約 22.6 GiB，超過 L4 的 22.03 GiB。Studio 會逐一按實際權重大小及畫布、Edit 參考圖尺寸估算用量，預留執行空間。每項任務都放得下且關閉 PE 時才使用 `--gpu-only`。含 BF16 的 L4 批次及有 PE 的批次會使用 `normal-vram`，讓 ComfyUI 可將部分權重移到系統記憶體。

L4 上使用這組 BF16 權重並關閉 PE 時，1024×1024 的 T2I 與雙圖 Edit 都通過 `normal-vram` 實測。

A100 已有單項 Q2_K Edit 實測。

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
  - 預設畫布依第 1 張參考圖的比例。未設定 `edit_resolution` 時，Job 的 `resolution: "2K"` 會使用約 4 MP，其他情況使用約 1 MP。設 `edit_resolution: 0` 會使用第 1 張圖的原始尺寸並對齊 32。可用 `"custom_size": true` 選擇自訂長寬比與解析度，或指定成對的 `width`、`height`。

---

## 4. 本地輕量 Web UI (`cstudio/webui.py` & `cstudio/static/`)

本專案內建由 **FastAPI + Uvicorn + 零建置原生 ES Module 前端** 驅動的雙欄式本地 Web UI（借鏡 Unsloth Studio 桌面版圖片生成介面）：
- **啟動方式**：`uv run cstudio ui [--host 127.0.0.1] [--port 7860] [--open]`（或 `uv run cstudio-ui`）。
- **部署範圍**僅限直接的本機 loopback 連線。`--host` 接受 `127.0.0.1`、`localhost`（綁定 `127.0.0.1`）與 `::1`（瀏覽器網址用 `[::1]`）。其他位址會在啟動前被拒絕。工作台信任能直接連線的本機使用者與程序。請勿透過反向代理、通道或連接埠轉送對外提供工作台。轉送器若讓請求看起來是直接的 loopback 連線，會繼承本機信任。
- 開啟首頁時，瀏覽器會取得工作台程序的 session cookie。API 與圖片需要該 cookie，程序重啟後須重新開啟首頁。控制請求須有與工作台協定、主機及埠相符的 `Origin`，並帶 `X-Colab-Studio-Request: 1`。會查詢 Colab 的狀態端點也需要此自訂標頭。工作台會拒絕代理轉送標頭。
- 請在取得 Google 登入網址的分頁貼回授權碼。登入網址有效 10 分鐘，每筆待登入流程只能交換一次。CLI 使用自己的待登入流程。提交後若交換失敗，請重新取得登入網址。成功登入會更新與 Colab CLI 共用的憑證。
- Web 改圖最多接受 10 張參考圖。上傳與畫廊沿用的圖片均限每張 35 MiB、單邊 8192 像素、總面積 16,777,216 像素。支援 PNG、JPEG、WebP、AVIF、GIF、BMP 和 TIFF，以首幀轉存 PNG。生成用 multipart 請求含表單內容共限 352 MiB，URL 編碼表單限 2 MiB。每個工作台程序同時處理一個生成請求的上傳與圖片轉檔。
- **核心功能**：
  - 頂部顯示登入、CU、Session 和本地待命期限。瀏覽器每 60 秒查詢雲端狀態，也可用「重新整理」主動查詢。自有 Session 的關機會區分已確認與待確認。
  - 左側提供 T2I、1～10 張有序 Edit 參考圖、RGBA 提示詞、Qwen 3.5 9B 擴寫、7 種長寬比與 Edit 的四種尺寸策略。
  - 右側顯示批次階段、日誌及原始與擴寫提示詞。遠端逐項完成後，圖片通過大小與 SHA-256 驗證才會發布到本地。歷史畫廊依本地 PNG 的可讀性與現有成果紀錄顯示，包含缺少成果紀錄的圖片。可下載原圖、沿用配方或加入 Edit 參考圖。
- **Python API 呼叫**：
  ```python
  from cstudio.runner import run_manifest, plan_batch
  from cstudio.colab_client import usage, sessions
  ```
  `run_manifest(..., progress_cb=my_callback, exit_on_error=False, should_abort=...)` 支援非 CLI 環境安全呼叫與中斷控制。

> `Qwen-Image-2.1-Turbo` 權重採用 Qwen Research License，使用範圍以授權原文為準。 [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE)
