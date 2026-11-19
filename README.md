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

# 4. 預覽範例批次任務
uv run cstudio plan jobs/demo_qwen21/manifest.json

# 5. 在 Colab L4 GPU 上執行並同步回 jobs/demo_qwen21/out/
uv run cstudio run jobs/demo_qwen21/manifest.json --gpu L4
```
