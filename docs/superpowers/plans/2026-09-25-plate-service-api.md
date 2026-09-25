# Plate Service API Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; user has requested direct implementation. Preserve the dirty worktree; no automatic commit/push.

**Goal:** 本機 1788 收圖片或 base64，經常駐 YOLO bbox + v3 OCR 回多牌 JSON。

**Architecture:** 新增隔離的 `plateai_service` package，不依賴 Studio、trainer 或舊 Reader。輸入／文字契約、推論管線、單執行緒容量管理與 FastAPI 分開；模型由明確 hash manifest 指定。

**Tech Stack:** Python 3.11, FastAPI/Uvicorn, Pillow, PyTorch cu118, Ultralytics, PaddleOCR 3.2.0。

**Spec:** `docs/superpowers/specs/2026-09-25-minimal-plate-recognition-service-design.md`

## Global Constraints

- 一次一圖；12 MiB image、13 MiB multipart、17 MiB JSON、20 MP、最多 20 牌。
- 127.0.0.1:1788；非 loopback 必須 token；不接受 URL／伺服器路徑；不存輸入圖片或辨識文字 log。
- 單 worker，模型同一執行緒載入暖機並推論；10 秒 timeout 後 native 工作結束前維持 busy。
- auto 分別初始化 detector / OCR GPU，失敗用相同模型 CPU 暖機；hash 失敗不能 fallback。
- 舊研究功能、release scaffold、active-v1 保留；本次先 API，不改 Studio UI、不散布權重。
- 格式 shape 提示不是法規；allocation / registration 不猜。軍牌與中文字頭回能力限制。

## Review Focus

1. Base64 / chunked / multipart body 限制不得被繞過。
2. 客戶端斷線或 timeout 不能釋放正在 native 推論的槽。
3. GPU fallback 後實際 device 要可觀察，模型損壞必須 fail closed。
4. EXIF 旋轉與 bbox 座標一致，未讀出仍保留 bbox。
5. 外部錯誤與 log 不包含 token、文字、圖片或本機路徑。

### Task 1: 輸入與文字契約

Files: `src/plateai_service/{errors,images,text}.py`, `tests/unit/test_service_contract.py`。
Interfaces: `decode_image(bytes, declared_mime=None) -> BGR ndarray`；`decode_base64(str) -> (bytes,mime)`；`read_plate_line(texts,polys) -> dict`。

- [x] 寫並執行失敗測試：`assert normalize_text(' ＡＢＣ－００１２ ') == 'ABC0012'`、中文字／未知符號保留、EXIF 尺寸、動畫拒絕、header pixels 限制、base64 MIME 不符、兩個合理牌號列 ambiguous。
- [x] 實作最小驗證與保守文字提取；執行 `python -m pytest tests/unit/test_service_contract.py -q`，13 passed。

### Task 2: 固定模型與管線

Files: `src/plateai_service/{config,pipeline}.py`, `tests/unit/test_service_pipeline.py`。
Interfaces: `ServiceConfig.from_file(Path)`、`verify_assets(config)`、`PlatePipeline(config).initialize()`、`.predict(bgr)`、`.metadata()`。

- [x] 寫失敗測試：hash 不符中止、20 牌上限、buffer clamp／排序、多牌 OCR 空白仍保留、GPU warmup 失敗 CPU 成功與兩者失敗。
- [x] 實作固定 v3 + 先 torch 後 paddle，沒有下載／模型掃描；pipeline 7 tests passed，含 review 後新增 GPU probe 失敗分支。

### Task 3: HTTP 與容量

Files: `src/plateai_service/{runtime,app,cli}.py`, `tests/unit/test_service_api.py`。
Interfaces: `create_app(config, pipeline_factory=None)`；lifespan 非阻塞開始載入；`/api/health`、`/api/ready`、`/api/predict`。

- [x] 寫失敗測試：file/base64/data URI 相同、統一錯誤、auth、body 超限、busy、timeout 後槽未釋放、fatal 撤 ready、init fail 仍 health。
- [x] 實作有限讀入與 in-memory multipart parsing，單 executor，shield native future、完成 callback 釋放 busy；API 13 tests passed。

### Task 4: 啟動與實測交付

Files: `run_api_1788.bat`, `run_api_1788.ps1`, `requirements/service-*.txt`, `docs/plate-service-api.md`, local ignored `runs/plate-service/`。

- [x] 提供獨立 Python 路徑啟動及固定模型 config；明確 setup 安裝，不在推論 pip/download。不改主 pyproject 依賴。實跑 root runner、PowerShell syntax checks、現有環境 install dry-run 無變動、pip check 通過；新機 bootstrap 尚未重建驗收。
- [x] 實際啟動後以同 bytes 三種輸入 POST，檢查多牌、health/ready、no-plate blank smoke；GPU 200 次交錯常駐請求、強制 CPU 16 圖功能回歸、修正後重啟 16 圖回歸均通過。16 圖文字集合與 Phase A v3 相同。
- [x] 最後完整 `python -m pytest -q`：522 passed / 1 skipped / 26 warnings，188.28s；skip 為缺少 user_cases.jsonl。`git diff --check` 通過；獨立 review 無已確認阻擋項，文件列明未驗證部分與重跑命令。未 commit／push。

## Progress / Rulings

- 開始：使用者明確要求繼續 API，依個人 AGENTS 指示直接實作，不再要求重複授權。工作留在既有 checkout 保留先前未提交更動，所有新邏輯在新 package。
- 本輪不接 Studio：獨立 API 可單獨驗收，避免把已 dirty 的研究 UI 與新服務混在一起。
- 不假造法規：文字 shape 辨識可顯示，法律 format / allocation 保守 unknown；後續有完整來源再收緊。
- Review：獨立 reviewer 未找到已證實 Critical/Important；33 個 focused tests 最終全過。指出 CPU forced 仍呼叫 GPU probe 的可強化處，將其視為 auto 退回契約邊界補強：新增失敗測試後修成 forced CPU 不 probe、auto probe 例外仍 CPU。未改其他推論選型。
- 尚不放行：真實 native GPU 初始化失敗環境的復原只用測試替身覆蓋；無 GPU 乾淨安裝、RTX 50、完整 50 圖品質、權重權利、cuDNN、斷網啟動與長期記憶體趨勢仍未驗收。不得以本機 API slice 代替整份 Phase B。
