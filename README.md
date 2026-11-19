# Colab Studio (`colab-studio`)

在自己的 **Google Colab GPU** 上遠端執行 `unsloth/Qwen-Image-2.1-Turbo-GGUF`（含官方 `Qwen 3.5 9B Prompt Enhancer` 文生圖／改圖擴寫模型），並自動將生成的圖片與擴寫提示詞同步回本地。全專案由 `uv` 管理。

詳細操作與 Agent 指引請見 [CLAUDE.md](CLAUDE.md)。

## 快速開始

```bash
# 1. 安裝 uv 環境並套用 colab-cli 補丁
uv sync && uv run cstudio-patch

# 2. 執行離線自我測試
uv run cstudio test

# 3. 檢查 Google Colab 登入與剩餘運算單元
uv run cstudio check

# 4. 啟動本地輕量 Web UI 工作台（支援文字生圖、1~10張多圖改圖、RGBA透明貼紙、9B AI擴寫與歷史畫廊）
uv run cstudio ui --open

# 5. 或使用 CLI 預覽與執行批次任務
uv run cstudio plan jobs/demo_qwen21/manifest.json
uv run cstudio run jobs/demo_qwen21/manifest.json --gpu L4
```
