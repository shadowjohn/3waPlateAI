# 最小可交付車牌辨識服務 — Spec

日期：2026-09-25　狀態：Phase A 小樣本實測完成；使用者已同意進 Phase B，先實作獨立 API，Studio 整合後續接入。

## 0. 分階段交付

- **Phase A（本次執行）**：在獨立 Python 環境驗證原生 YOLO＋PaddleOCR，固定 checkpoint、圖片與裁切參數，跑一小批完整場景。比較 PP-OCRv3／PP-OCRv5 的 mobile 偵測＋英文辨識模型；同圖、同 YOLO ROI、同執行裝置／執行緒及前後處理，關閉文件方向、文件拉正與文字列方向分類。保存原始文字、每牌結果、裁切、裝置、冷啟動與暖機後延遲，以及來源／套件／模型 hash。這是可丟棄的本機 probe，不是正式服務，也不是新的訓練實驗。
- **Phase B（已獲同意）**：依 Phase A 結果先固定 v3 mobile，實作下述 1788 API；1688 整合與完整品質驗收接續進行，不啟用／替換現役模型。
- 小樣本以手邊已授權圖片為限，先凍結名單及人工文字 GT 再跑 OCR；既有看過的照片只算探索／回歸。真實無牌、多牌、軍牌、古董牌等不足時明列未覆蓋，不以合成圖或一般車牌冒充。Phase A 的本機推論時間不等於 Phase B 的 HTTP 延遲。
- **PaddleOCR 套件 3.x 與 PP-OCRv3／v5 模型代際是不同概念**。比較明確模型名稱，不拿 v3 mobile 與 v5 server 的差異宣稱世代差異；不預設 v3 必然快或 v5 必然準。

## 1. 要交付什麼

> 別人問我們有沒有車牌辨識功能，我們能給他 API 網址；他傳一張圖，就拿到準確、快速、可供專案使用的 JSON 結果。

第一版服務對象是既有專案的程式開發者。支援單張照片中的多張台灣汽／機車單列英數車牌；不是要求對方先裁切、人工點角或操作研究工具。1688 提供同一服務的可視化試用，1788 可獨立運行，不依賴 Studio 開著。

交付主指標是整牌文字正確率、定位漏檢／誤報、端到端延遲與服務穩定性。四角誤差不再是服務上線的必要門檻。

## 2. 範圍與取捨

| 第一版做 | 第一版不做 |
| --- | --- |
| YOLO 定位、bbox 加 buffer、PaddleOCR | 新增訓練、追求四角貼齊、改 backbone |
| 模型啟動載入、暖機、常駐重用 | 不停機換權重、模型管理平台 |
| GPU 優先、不可用時自動退回 CPU | 強制 ONNX 匯出、TensorRT／量化優化 |
| 檔案／Base64 輸入、單張多牌、JSON、1688 畫框與列表 | 預設 A/B 雙管線、多 OCR 投票 |
| 健康／就緒檢查、容量限制、明確錯誤 | 分散式佇列、批次工作系統、串流追蹤 |
| Windows 可重現啟動、實際 API 驗收 | Docker／Kubernetes、帳務或多租戶平台 |

「熱載」在本版專指**服務啟動載入並暖機、請求之間重用模型**。換模型以修改部署設定後重啟處理，不偷渡不中斷切換機制。

翻正、透視拉正、亮度／對比調整保留為後續針對實際 OCR 失敗案例的改善；第一版不預設開啟文件拉正、方向分類、二值化、銳化或 OpenCV 四角精修。

後續有實際改善、需要整合既有 PHP 專案時，再考慮 `api.php?mode=tw_plate_ocrv&version=1` 入口：固定 `mode=tw_plate_ocrv`，版本另以 `version=1`、`2`、`3`、`4` 選擇，不把版本號接在 mode 名稱後。這是延後的服務版本／整合方式，不是本版要實作的 PHP wrapper、版本分流或多版本管理平台；先交付一個能穩定傳圖回 JSON 的服務。日後新增版本時保留既有呼叫契約，不讓已接入的專案被迫同步升級。

## 3. 現有程式與最小架構

盤點依據：`src/plateai_web/app.py`、`predictor.py`、`external_predictor.py`、`mit_lpr_crop_api.py`、`release_packager.py` 及 `release/3wa_plate_api/README.md`。

- 1688 已有上傳、多牌回應、畫框、辨識列表與模型快取，可沿用 UI，不另造研究頁。
- 現有 `/api/predict/compare` 會執行 active 與 candidate 兩套推論，不用於新的預設服務路徑。
- 1788 目前是 source-only 雛形，尚未具備完整實拍辨識能力；不能把目前目錄當作已完成的服務。
- 現有 MIT crop API 只有裁切圖 OCR，不能冒充 full-scene API；保留原功能。

```text
其他專案 ──────────────────────────┐
                                  ↓
1688 上傳／貼圖 → 固定目的地代理 → 1788 /api/predict
                                  ↓
                         常駐 YOLO → buffered ROI → 常駐 PaddleOCR
                                  ↓
                         每牌文字、bbox、狀態、耗時 JSON
                                  ↓
                      呼叫方自行處理／1688 畫框與裁切預覽
```

1788 是新路徑唯一的推論程序。1688 用現有 HTTP client 依部署設定呼叫它，不再載一份相同模型；也不為單次操作先跑舊模型。舊 A/B 與研究功能保留為明確選用項目，不更改舊 endpoint 的回應格式。

實作維持三個簡單邊界：固定推論管線、薄 FastAPI 包裝、Studio 代理與呈現。不建立可插拔演算法框架。服務原始碼放在 `src/`；發行入口引用同一份實作，不把完整服務邏輯複製進 `release_packager.py` 字串。

## 4. 推論契約

1. 檢查輸入、套用 EXIF 方向並解碼一次。回傳的影像尺寸與所有座標都以這張方向校正後的完整影像為準；預覽亦須一致。
2. YOLO 在固定 640px letterbox 輸入定位，將 bbox 還原到完整影像座標；保留 NMS 後所有達門檻的牌，不能只取 top-1。
3. 初始設定：detector confidence `0.25`、NMS IoU `0.45`；每邊增加 bbox 寬／高的 `5%` 作為 buffer，裁切不得越出影像。這些是待實測的固定起始值，不是最佳化結論。
4. 在原圖解析度裁出彩色 ROI，交由 PaddleOCR 的文字偵測＋辨識處理，不把整塊含標語、牌框的 ROI 直接當成單一文字行。Phase A 比較 `PP-OCRv3_mobile_det`＋`en_PP-OCRv3_mobile_rec` 與 `PP-OCRv5_mobile_det`＋`en_PP-OCRv5_mobile_rec`；Phase B 依實測選定並固定其中一組，不追隨上游預設模型變更、不預設每張跑兩套。
5. 同一文字列的片段由左到右組合；不把不同列的標語與牌號串在一起。僅作下述保守正規化，不猜補字、不自動把 O 改成 0。格式符合只表示命中已知編碼規則，不代表已核發、真牌或合法上路。
6. 每個 detector 候選都有結果；OCR 空白、未知格式或單牌失敗不能讓框消失。多個合理牌號無法唯一選定時回 `ambiguous`，不用 GT 或最高分硬選答案。

每張最多處理 20 個 NMS 候選以限制成本；多於 20 個時回 `422 too_many_candidates`，不靜默截斷成「全部結果」。初版逐牌處理即可，批次 OCR 只在量測顯示需要時加入。

YOLO 可先用現有微調前的 YOLO11n-pose 開發 checkpoint，只消費 bbox，不使用其角點。這是本機開發起點，不是模型已通過發佈審核；也不啟用此次 clean83 微調 checkpoint。實際部署必須指定模型檔與 SHA-256，不能掃描目錄自動取 latest。

### 文字正規化、特殊牌與異常示警（Phase B 契約）

- `raw_text` 保留選中牌號列的原始 OCR 字串；`text` 僅統一全形 ASCII 英數、英文字母大寫、空白及已知分隔符，保留中文前綴、前導零與所有其他可見字元。不得用「只留下 A-Z0-9」清掉軍／使／外等身分字，也不得做 O/0、I/1、B/8、S/5 猜字修正。未知符號保留並示警。
- 新增 `display_text`：只有格式與分隔位置可唯一確定才補標準連字號；否則為 null，不猜造顯示格式。既有 `text` 不帶裝飾性連字號的契約保持不變。
- 新增 `format_check`：包含 `status`（`matched`／`unknown`／`ambiguous`／`mismatch`）、`rule_id`（可 null）、`ruleset_version`、`allocation_status`（`within_documented_range`／`outside_documented_range`／`unknown`）及 `registration_status: "not_checked"`。`warnings` 是帶 `code`／`message` 的陣列；無警告回空陣列。示警不改文字、不丟掉 bbox。
- Phase B 第一個 API 切片：`recognized` 僅表示單一 OCR 文字符合保守英數形狀，非官方編碼驗證；官方規則尚未完整載入時 `format_check.status` 與 `allocation_status` 均回 `unknown` 並帶提醒，`rule_id` 為 null。`display_text` 的連字號僅按唯一文字形狀分段，不能視為官方標準化結果。後續官方編碼查核與 OCR 提取成敗分開，不先假造通過結果。
- 使用車種／年代分開的官方規則，不把新式 I/O/4 限制套到全部舊牌。既有 `tw_standard_v1.json` 是 OCR／合成格式集合，並非完整法規或實際發牌資料庫；不得直接升格為合法性判定器，也不為新服務改壞既有訓練字集。
- 古董車：交通部 2023-06-14 公告的專用號牌是棕底白字，古董汽車 `KUA`、古董機車 `KUM`，公告數字範圍 0001–9999、各 6560 副，須連同新式編碼排除規則使用，不能把範圍內每個整數都當成可發牌。前綴僅提供「符合古董牌編碼」提示，不證明車齡、車籍或當下行駛許可。[官方公告](https://eycc.ey.gov.tw/Page/9FAC64F67005E355/d26f009b-15c5-4a9e-a1dd-967751ee140a)
- 軍牌／外交牌等中文或特殊字頭：本次尚未取得完整且現行的官方編碼表，不杜撰 regex 或套用一般民牌字數。標記 `special_plate_unverified`。目前英文 recognizer 未驗證中文識別，不能宣稱支援完整軍牌；中文模型／實拍樣本另列能力缺口，不靠模型漏字後剩餘英數判定普通牌。
- 已知範圍與實際核發分離：官方「預計分配範圍」不等於已發出的有效車籍。規則資料記錄來源、公告日期、生效期間（未提供則 null）、擷取日期與版本；106 年對照表只作歷史資料，不假裝現行全集。資料過時、類別未知、年代不明或規則衝突時回 `unknown`，不發確定性違規警報。
- 首版警告包括 `format_unknown`、`format_ambiguous`、`unexpected_character`、`allocation_outside_known_range`、`allocation_reference_stale`、`special_plate_unverified`、`ocr_charset_limited`。只有有適用且完整的版本範圍依據才能發 outside 警告；未知不是異常。UI 使用「請複核／不符合已載入規則」，不顯示「假牌／贓車／違法」，亦不新增車籍查詢或黑名單服務。

## 5. 對外 API

### `POST /api/predict`

同一 endpoint 提供兩種輸入，走相同驗證、推論管線與回應契約：

| 輸入方式 | Content-Type | 圖片欄位 |
| --- | --- | --- |
| 檔案上傳 | `multipart/form-data` | `file`：圖片檔案內容 |
| Base64 | `application/json` | `image_base64`：標準 Base64 字串，或 `data:image/jpeg;base64,...` 等 data URI |

每次只收一張、一種輸入；缺少圖片或同時提供 `file` 與 `image_base64` 都回明確錯誤，不猜優先順序。Base64 嚴格驗證格式並限制解碼前長度；data URI 僅接受對應 JPEG／PNG／WebP 的 MIME，仍須核對實際影像格式，不能只相信宣告。解析輸入後，兩種方式都使用相同圖片 bytes／pixel 限制。

第一版只接受 JPEG／PNG／WebP 單張靜態影像，不接受遠端 URL、伺服器檔案路徑、模型路徑或批次 ZIP。「檔案」是呼叫方上傳的內容，不是要求服務讀取本機路徑。

成功回應示意（數值僅為契約範例，不是量測成果）：

```json
{
  "schema_version": 1,
  "request_id": "server-generated-id",
  "status": "ok",
  "image": {"width": 1920, "height": 1080},
  "model_id": "configured-model-set-id",
  "detections": [
    {
      "id": 1,
      "bbox": [420.0, 610.0, 610.0, 710.0],
      "crop_bbox": [410, 605, 620, 715],
      "text": "EAB5555",
      "raw_text": "EAB-5555",
      "status": "recognized",
      "reason": null
    }
  ],
  "timings_ms": {"decode": 3.0, "detect": 20.0, "ocr": 40.0, "total": 65.0}
}
```

- `bbox` 是未加 buffer 的原圖 `[x1,y1,x2,y2]` 浮點 pixel 座標；`crop_bbox` 是實際裁切用整數座標，左上向下取整、右下向上取整並裁限，右／下界為 exclusive。兩者不混用。
- 同一次回應依 bbox 左上 y、x 排序後編號，JSON、畫面框號與列表須一致。
- `text` 是正規化後的辨識文字，無法唯一給值時為 `null`；`raw_text` 保留選中牌號列的原文，無單一候選則為 `null`。
- 每牌 `status`：`recognized`、`unreadable`、`unverified_format`、`ambiguous`、`error`。後四者的 `reason` 分別使用 `ocr_empty`、`format_unknown`、`multiple_text_candidates`、`crop_invalid`／`ocr_failed`；未知格式可保留非空 `text`，但不計入已確認辨識成功數。
- 頂層 `status`：全部候選 recognized 為 `ok`；有任一牌未確認為 `partial`；沒有 detector 候選為 `no_plate` 並回空陣列。三者都是 HTTP 200，不把「無車牌」當系統故障。
- 不預設附原圖、base64 圖片或 OCR 信心百分比。1688 用上傳圖與 `crop_bbox` 產生裁切預覽；API 保持小而直接。
- `total` 是服務端處理耗時；真正客戶端端到端耗時另外量測，不能把它當成同一件事。

### 健康、就緒與錯誤

| 端點／情況 | 回應 |
| --- | --- |
| `GET /api/health` | 程序存活回 200；只代表 alive |
| `GET /api/ready` | 模型載入、雜湊檢查與暖機成功回 200；否則 503 |
| 受保護端點缺少／錯誤 token | 401 `unauthorized` |
| 未提供圖片 | 422 `missing_image` |
| 同時提供 file 與 image_base64 | 422 `ambiguous_image_input` |
| JSON 或 Base64 格式無效 | 400 `invalid_json`／`invalid_base64` |
| 損壞／空影像 | 400 `invalid_image` |
| 不支援的格式或動畫 | 415 `unsupported_image` |
| 超過 12 MiB 或解碼後 20 MP | 413 `image_too_large` |
| 超過候選上限 | 422 `too_many_candidates` |
| 未就緒 | 503 `model_not_ready` |
| 已有推論佔用容量 | 429 `busy`，附 `Retry-After: 1` |
| 推論逾時 | 504 `inference_timeout` |
| 全域推論失敗 | 503 `inference_failed`，不冒充 no_plate |

錯誤統一為 `{"request_id":"...","error":{"code":"...","message":"..."}}`；驗證錯誤亦使用此格式。外部不回 traceback 或本機模型路徑。`/api/ready` 提供 model_id、各模組的 `requested_device`／`effective_device`、`fallback_reason`、runtime 版本與暖機狀態，不宣稱辨識準確率。

單一 ROI 的資料／裁切錯誤可以形成 partial；runtime、裝置或模型本身不可用屬全域失敗，應回 503 並撤下 ready，不能把整個 OCR 壞掉包裝成多筆正常未讀出。

## 6. 常駐、容量與最小部署保護

- 單一服務程序、一個推論執行緒、一份模型實例；載入與暖機不阻塞健康端點。載入失敗保持 not-ready，記錄原因，不暗自換舊模型。
- 啟動時分別暖機 detector 與 OCR（用固定非敏感測試 ROI，不能因空白圖沒有 bbox 而跳過 OCR）。只有兩者都完成才 ready。
- 裝置預設 `auto`：detector 與 OCR 各自優先嘗試可用 GPU；沒有相容 GPU、GPU runtime 不可用，或 GPU 初始化／暖機失敗時，改用相同模型的 CPU 路徑重新暖機。CPU 成功即可 ready，允許 GPU／CPU 混合；記錄退回原因並在 1688 服務狀態顯示實際裝置，不把退回視為無法服務。另可設定 `cpu` 強制 CPU 以利部署與驗證。
- 裝置選擇在啟動階段完成並常駐重用，不逐張探測／重建。第一版 GPU 支援以 NVIDIA CUDA 為準，其他尚未驗證的 GPU 使用 CPU。不能只看見顯卡就宣稱 GPU 推論；須完成模型載入與暖機。模型檔損壞、雜湊錯誤或 CPU 路徑也失敗仍保持 not-ready，不用裝置退回掩蓋模型問題。運行中的致命推論錯誤依既有 503／撤下 ready 契約處理，首版不做自動重建與逐請求 GPU 重試。
- 推論在 event loop 外執行；第一版不設無界等待佇列。忙碌時快速回 429；容量控制須在昂貴解碼／推論前生效，非推論路由維持可回應。
- 推論預算 10 秒。HTTP 逾時或客戶端斷線不代表 native inference 已取消：工作未結束前不得釋放忙碌槽或啟動另一份模型。無法恢復時由操作者重啟，首版不做 worker 自動重建。
- 有限量串流讀入上傳：圖片 bytes 上限 12 MiB；multipart 整個 body 上限 13 MiB；Base64 JSON 整個 body 上限 17 MiB（容納 Base64 約 4/3 的編碼膨脹與 JSON／data URI 開銷）。Base64 payload 本體上限 16 MiB，解碼後仍不得超過 12 MiB。兩條路徑都在像素解碼前檢查影像 header 的 20 MP 限制，不能先無上限 read／完整解碼再判斷過大。1688 代理採相同上限。
- 預設 `127.0.0.1:1788`、單 worker、關閉開發 reload。1688 代理目的地只由伺服器設定提供，不接受使用者任意指定 URL。
- 要開放內網，需明確設定 bind 與 API token；非 loopback 部署缺 token 應拒絕啟動。predict、ready 及 API 文件需驗證 token；health 只公開存活狀態。1688 的代理也要保留相同存取限制，不能藉未保護的 Studio 繞過驗證。
- CORS 預設關閉，1688 走同源代理。跨網路存取由既有 HTTPS reverse proxy 保護，不直接把裸 1788 曝露到 Internet；不新增帳號管理平台。
- 圖片、裁切、牌號預設不落盤、不寫 access log；只記 request_id、狀態、張數、耗時與模型 ID。token 不送到可下載的前端設定或寫入 Git。

## 7. 1688 使用體驗

沿用 Live Inference 的拖曳、上傳、貼圖與預覽。新增的服務模式在完成驗收後成為預設；舊候選／A/B 明確留在進階選項，不破壞既有研究工作。

- 上傳一次只送出一次新服務請求，顯示 loading／忙碌／未就緒；不背景輪流跑多個模型，也不自動無限重試。
- 原圖顯示所有候選 bbox 與編號；右側一牌一列，顯示文字、bbox、實際 buffered crop 與狀態。不要求使用者看研究診斷才能找到結果。
- 讀不出的框仍可點選查看裁切；顯示「找到牌、未讀出」等明確狀態，不產生假牌號。
- 每次新上傳清空前次結果；舊請求晚到不能覆蓋新圖片。前端載入的 EXIF 方向與 API 座標保持一致。
- 頁面顯示整次耗時、服務狀態與 detector／OCR 實際使用的 GPU 或 CPU；有退回時顯示簡短原因。不預設顯示 OCR 信心百分比或 A/B 表格。

## 8. 啟動、依賴與發行邊界

沿用 Python 3.11／FastAPI／Uvicorn／httpx。第一版直接使用 YOLO 原生 PyTorch 與 PaddleOCR 原生推論；放在獨立服務環境，提供 Windows `run_api_1788.bat`／PowerShell bootstrap，不讓其依賴解析改壞現有訓練環境。

實作先做一次相容性預檢，將確實可載入兩套模型的套件版本寫入服務 lock；本規格不假造尚未測試的版本組合。先驗證這台 GTX 1080 的 GPU 優先路徑與無 GPU 的 CPU 路徑，不把 PaddleOCR 固定在 CPU。是否能用 GPU 取決於該模型與 runtime 的實際相容性；不可用即依上述 auto 規則退 CPU，不能要求每台部署機器都有 NVIDIA 顯卡。setup 負責檢查與安裝相容依賴，服務不在推論時執行 pip 或切換套件版本。

PyTorch 安裝 profile 依使用者指定固定分開：GTX 1080 使用 `cu118`（本機已用過 `torch 2.7.1+cu118`）；RTX 5060／5090 使用 `cu128`；無 GPU 使用 CPU profile。版本需搭配相容 torchvision 並固定 lock；不在同一環境來回升降、不用 nvidia-smi 顯示的 CUDA 上限當作已安裝的 torch wheel 版本。官方 PyTorch 2.7.1 提供 Windows cu118／cu128 wheel，Blackwell 支援由官方 2.7／cu128 說明確認；5060／5090 仍需在對應實機驗證，不能以 1080 測試代替。Paddle 有自己的 runtime 相容條件，不能由 PyTorch cu118／cu128 成功推定 Paddle GPU 成功。[官方版本表](https://pytorch.org/get-started/previous-versions/)／[Blackwell 說明](https://pytorch.org/blog/pytorch-2-7/)

**ONNX 留作後續可選部署優化，不是第一版完成條件。** 原生管線先通過服務驗收；之後若轉 ONNX，須逐模型核對前處理、字典、輸入／輸出契約、bbox 與 OCR 字串的一致性，並量測 GPU／CPU 延遲與依賴成本。有匯出檔不等於可替換，也不預設 ONNX 一定比較快；通過驗證再切換，API 保持相同。

依賴與模型取得是明確的 setup 步驟，不在每次啟動／推論時追 latest 或下載。每組模型記錄來源、固定版本／revision、hash 與使用限制；配置完成後拔除外網也須能啟動辨識。README 附 `curl.exe -F "file=@car.jpg" .../api/predict` 檔案上傳與 JSON `image_base64` 兩種可執行範例、回應說明、啟停方式與 readiness 檢查即可完成基本接入。

本專案尚未同意因 YOLO 改變專案授權。現有 EZCon 衍生 checkpoint 是本機研究權重，不能因寫了 API 就當作可供客戶使用或散布的模型。**可供目標用途的模型與 runtime 條件，是發佈前必須完成的一次審核**；未完成時只交付本機測試版，明確標示，不宣稱可正式部署。保留原有 source／模型分離及 notices，不把整個 `runs/`、dataset 或研究權重塞入 release。

## 9. 驗收：用 API 證明，不用研究分數代替

以下數值是供本次 spec 審閱的首版目標，並非已達成的成績。

### 功能與品質

- 從 repository 外的另一個程式實際 POST 圖片取得 JSON；1788 單獨啟動亦可完成。API 與 1688 同圖同模型結果一致。
- 同一份圖片 bytes 分別用檔案、純 Base64、Base64 data URI 呼叫，除 request_id／耗時外結果須一致。覆蓋壞 Base64、空值、混合輸入、超長編碼、解碼後超限與 MIME 不符，不得讓 Base64 繞過圖片限制。
- 固定至少 50 張完整場景實拍：至少 10 張真實多牌、10 張無牌，總計至少 50 個可讀 GT 車牌；另含斜牌、明暗差異與讀不出的個案。不是只用已裁好的 3,000 張 OCR 圖驗收定位。
- 推論前固定 manifest、圖片 hash、整張圖的所有車牌 bbox 與文字；可讀性也須預先標記。實拍資料與模型訓練來源的關係如實記錄；既有反覆看過的 36 張只當回歸例，不稱新盲測。合成多牌可作功能測試，但不能取代真實多牌品質驗收。
- 以 bbox IoU ≥0.5 作一對一 GT 匹配，重複框算 FP。定位 precision／recall 各 ≥95%；可讀 GT 上的端到端整牌 exact match ≥95%（漏檢與未讀出仍在分母）。無牌組不得產生 recognized 牌號。這是限定場景小樣本驗收，不外推所有場景準確率。
- 每張牌去除空白、統一大小寫與分隔符後比文字；不套 O/0 等猜字修正。報告全部樣本及失敗圖，不只展示成功圖。
- 正規化測試另含全形英數、前導零、中文前綴、未知符號、舊牌 I/O/4、新式排除字元、KUA／KUM、過時範圍及格式多解；測試字串只證明契約，不代替特殊牌的實拍 OCR 驗收。格式／配號提示與整牌 OCR exact match 分開計分，不把通過 regex 當文字正確。

### 回應速度與穩定性

- 在 GTX 1080 的 GPU 優先設定、同機 HTTP、最大 1920×1080 JPEG 上，暖機後單牌 P95 目標 ≤500ms；四牌 P95 目標 ≤1,500ms。逐模組記錄實際 device，混合 GPU／CPU 不能標成全 GPU。純 CPU 路徑須有同樣的功能與品質驗收，P50／P95 另外實測公布，不套用 GPU 延遲目標；沒有 GPU 不得因此拒絕啟動，既有單次推論 10 秒預算仍適用。
- 單牌／四牌各固定 10 張實拍、每張 10 次，共各 100 次 HTTP 請求，報告客戶端完整 P50／P95 及各階段時間；冷啟動到 ready 時間另列，不混入暖機後統計。這是延遲回歸，不是準確率獨立樣本數。
- 至少連續 200 次混合請求不重建模型、無未處理錯誤、無逐次累積佔用；回應與真實模型初始化計數共同驗證，不只看某一次很快。
- 覆蓋 GPU 可用、無 GPU、GPU 初始化／暖機失敗轉 CPU、強制 CPU 及兩者都失敗。CPU 退回後只建立並重用常駐實例，不逐請求重試 GPU；ready／UI 的實際裝置與 log 一致。GPU 與 CPU 都跑同一份固定品質清單，記錄數值／結果差異，不假設兩者逐位元相同。
- 兩請求同時進入時不併發操作同一模型；容量外請求按契約快速拒絕，health 仍可回應。用測試覆蓋逾時後工作仍執行、壞圖、超大圖、缺模型、單牌失敗與服務中斷。
- 未達品質或適用的速度目標，列明失敗項並先定位階段瓶頸，不自行新增一串前處理、偷換模型或宣稱「可發服務」。明確標示的 CPU 退回是預期部署方式，不等於服務故障，也不能冒用 GPU 的量測成績。

### 交付清單

一個可重現啟動的 1788 服務、一份固定 API 契約、一個 1688 可視化入口、一頁接入文件、一份實測結果。模型權利、基本保護、功能、品質與延遲目標通過後，才稱為約定部署場景的可交付版本；不把本機 smoke test 叫作公開服務已驗收。

## 10. 參考與下一階段

來源索引維持在 [reference.md](../../../reference.md)，不把實驗結果塞進參考資料文件。[PaddleOCR 官方流程文件](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md) 用於核對模組與模型名稱，不引用其官方 benchmark 充當本服務數字。

使用者檢視 Phase A 報告後已同意繼續 API。第一個交付切片是本機獨立 1788 API（v3、bbox 加 buffer、不做旋轉拉正）；Studio 代理／預設入口、完整 50 張驗收、權重授權放行仍是後續工作，不把本機 API 跑通等同完成發行驗收。格式檢查先明確區分保守的文字形狀提示與未驗證的法規／配號：不載入假造的現行編碼全集，不發 outside-range 或違法警報。
