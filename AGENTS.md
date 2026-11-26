# Colab Studio 維護指南

## 入口與架構

- `cstudio/cli.py` 是 CLI 入口。`cstudio/webui.py` 與 `cstudio/static/` 提供本機工作台。
- `cstudio/runner.py` 共用於 CLI 與 UI，負責 manifest、指紋、雲端協調和成果同步。`cstudio/colab_client.py` 管理 Session、keep 與 watchdog，`cstudio/_colab.py` 封裝釘選版 Colab CLI，`cstudio/colab_login.py` 管理 PKCE 登入。
- `remote/remote_runner.py` 在 Colab VM 內呼叫 ComfyUI。`remote/models/` 是模型選取與 graph 的共用來源。`remote/versions.json` 和 `schema/manifest.schema.json` 分別記錄版本與 manifest 格式。
- 操作方式和模型背景參考 [CLAUDE.md](CLAUDE.md)。功能與相容性以目前程式、schema、版本表及測試為準。

## 維護與驗收原則

- 每批開工前說明要驗證什麼、如何判定完成，以及最多嘗試幾次。達到完成條件就收尾，達嘗試上限仍未通過就停止並記錄結果。
- 將確認的模型或平台限制寫入文件，標明適用條件及已驗證範圍。產品行為已驗證時，以已知限制結案。
- 優先使用 Studio 的 CLI、工作台和既有測試。可重現的產品缺陷用精簡的回歸測試覆蓋，避免為單次驗證另建一套測試工具或控制器。
- 連續兩批都沒有 repo 變更時，先與使用者檢討方向和已取得的證據。檢討完成前，暫停核准及執行下一批。
- 延後項目見 [ROADMAP.md](ROADMAP.md)，只在觸發時機出現時處理。

## 開發與離線驗證

- 使用 Python 3.12 與 uv。建立開發環境時用 `uv sync --locked` 依 `uv.lock` 安裝。
- 在既有環境執行與 Linux CI 一致的離線基準。[GitHub Actions 工作流程](.github/workflows/ci.yml)

```bash
uv run --locked --offline --no-sync python -m unittest discover -s tests
uv run --locked --offline --no-sync cstudio test
node --test tests/test_webui_frontend.cjs
ruff check cstudio remote tests
node --check cstudio/static/app.js
uv lock --check --offline
git diff --check
git diff --cached --check
```

- `uv run --locked --offline --no-sync cstudio plan jobs/demo_qwen21/manifest.json` 可離線規劃批次。回歸測試使用暫存目錄、合成資料、Colab 替身及可控時鐘。
- `cstudio-patch --status` 會唯讀檢查目前虛擬環境中的 Colab CLI。執行 `cstudio-patch` 會修改此虛擬環境的釘選套件，Studio 子程序已在程序內停用更新檢查。追查補丁時先用套件測試樣本驗證，實際安裝依部署需求。

## 需保護的行為

- 在配置 GPU 前完成輸入驗證。改動相關流程時，覆蓋正常重用、續跑、強制重做及確認後釋放 GPU。
- 成果記錄需能對應 run、job 與驗證狀態。下載使用獨立暫存檔，驗證後才發布。重做和中斷時保留可讀的舊成果與回復依據。
- 自動回收限身分可驗證的自有 Session。核對目前 OAuth 授權與 runtime endpoint，保留既有 `ReleaseState` 契約，區分結果未知、等待確認和已確認釋放。
- 保持 1～10 張有序 Edit 參考圖、T2I 與 Edit 的 PE、RGBA、尺寸策略，以及 INT8／BF16／GGUF 的相容性。若核准範圍涉及模型、量化、節點、版本或推論參數，依上游與測試證據驗證相容性。
- 對 Web 控制、檔案、影像和 OAuth 使用可測的信任邊界。請求錯誤與日誌要遮罩授權碼、OAuth／runtime token 和 PKCE verifier。

## Colab 與資料界線

- 保護現有 `jobs/`、圖片、results、credentials 及既有未提交差異。測試用獨立暫存工作區保存 manifest、試片、results 和 keep。
- 真實 Colab 操作、OAuth、GPU 測試或大型模型下載須另行取得使用者核准。會耗用 CU 的測試先確認目標和預算，實測後核對指定 runtime 與清理結果。

## 修改與交接

- 接手時先執行 `git status --short` 和 `git rev-parse HEAD`。版本已往前時，讀取新增差異後再接續。
- 每批聚焦使用者核准的項目。以能覆蓋缺陷的回歸測試驗證關鍵失敗路徑，並跑既有離線檢查。回報結果、Git diff 摘要及剩餘風險。
- Commit 和 push 依使用者逐批授權。獲准直接推送時沿用既有 `main`，無須另開分支或 PR。提交使用 Conventional Commits，Colab 修復沿用 `fix(colab): subject`。
