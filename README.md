# Colab Studio (`colab-studio`)

在自己的 **Google Colab GPU** 上執行 Qwen-Image-2.1 Turbo 文生圖與多圖改圖。Studio 預設使用原生 INT8，可選 BF16 或 Unsloth GGUF，並可啟用 Qwen 3.5 9B 提示詞擴寫。成功完成的圖片會經大小與 SHA-256 驗證後同步回本地。

詳細操作與 Agent 指引請見 [CLAUDE.md](CLAUDE.md)。

## 快速開始

```bash
# 1. 使用 Python 3.12 和 uv 安裝鎖定依賴
uv sync --locked
uv run --locked --offline --no-sync cstudio-patch

# 2. 執行離線自我測試
uv run --locked --offline --no-sync cstudio test

# 3. 查詢目前安裝的補丁狀態，再檢查 Colab 登入與剩餘 CU
uv run --locked --offline --no-sync cstudio-patch --status
uv run cstudio check

# 4. 啟動本機工作台，可在工作台完成 Colab 登入
uv run cstudio ui --open

# 5. 或使用 CLI 預覽與執行批次任務
uv run --locked --offline --no-sync cstudio plan jobs/demo_qwen21/manifest.json
uv run cstudio run jobs/demo_qwen21/manifest.json --gpu L4
```

## 驗證範圍

Colab L4 搭配 Q2_K 已實測 T2I、RGBA、提示詞擴寫（PE），以及 1～10 張 Edit 參考圖的有序傳輸、SHA-256 驗證、ComfyUI history 和成果同步。參考圖較多時，尤其是 9～10 張，構圖、編號和挖孔位置受模型及量化限制，Studio 不保證精確重現。

L4 上的原生 INT8 與 Q4_K_M 已通過 T2I 和雙圖 Edit。INT8 也驗證了 768×512 的 Edit 首圖尺寸及 2048×2048 的 2K T2I。BF16 的 T2I 和雙圖 Edit 在本批混合權重、L4（22.03 GiB）、GPU-only、PE 關閉的條件下因 VRAM 不足而失敗。

Linux 已實測 OAuth、CLI 和本機工作台。macOS、Windows 與 Windows ACL 未驗證。GPU 的實際可用性取決於 Colab 帳號、配額與配置。[操作與驗證範圍](CLAUDE.md#2-模型設定與驗證範圍)
