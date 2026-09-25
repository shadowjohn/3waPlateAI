# 車牌辨識 API（本機測試版，1788）

目前管線：**常駐 YOLO bbox → 每側 5% buffer → 常駐 PP-OCRv3 mobile + 英文辨識**。不做四角校正、翻正、二值化；不執行舊 A/B。不需要開著 1688。

這是可實際傳圖的本機 API，不是完成授權／品質放行的客戶發行版。研究 checkpoint 與 Ultralytics 的授權、資料衍生權重用途仍須獨立確認。`release/3wa_plate_api` 舊 source-only 雛形沒有被改寫，也不是新服務入口；1688 目前尚未代理此 API。

## 本機啟動

本次已測環境：Windows / GTX 1080 / Python 3.11.15 / torch 2.7.1+cu118 / Paddle GPU 3.2.2 / PaddleOCR 3.2.0。

現有隔離環境可直接啟動，不安裝至主專案 `.venv`：

```powershell
.\run_api_1788.ps1 -Python 'D:\mytools\3waPlateAI-experiments\plate-service-phasea-20260925\env\Scripts\python.exe'
```

預設使用本機未納入 Git 的 `runs/plate-service/config.json`。此檔明確記錄 checkpoint、兩個 OCR 模型目錄與每個檔案的 SHA256，不掃描 latest；檔案不存在／hash 不合則 not-ready。只使用你信任且有權使用的模型檔；hash 完整性不是來源可信度審核。

可先設定 `$env:PLATEAI_SERVICE_PYTHON`，再雙擊根目錄的 `run_api_1788.bat`。按 Ctrl+C 停止。單 worker、無 reload；模型換檔後須更新可信 manifest 並重啟，不做不停機換權重。

新機器可執行 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_api_1788.ps1 -Profile cu118`，建立獨立 `.venv-service`（已有目錄時拒絕覆寫）。需要預先安裝 uv。腳本不下載資料或模型，模型與 manifest 另行提供。

Linux 同樣需要 `uv` 與 CPython 3.11，並使用獨立 `.venv-service`：

```bash
chmod +x setup_api_1788.sh run_api_1788.sh
./setup_api_1788.sh --profile cu118  # 或 cu128、cpu
./run_api_1788.sh --config /secure/path/plate-service.json
```

預設只綁定 `127.0.0.1:1788`，可供同機展示頁呼叫。`PLATEAI_SERVICE_PYTHON` 可指定既有的隔離環境；`--python`、`--device auto|cpu`、`--host` 與 `--port` 對應 PowerShell 啟動參數。若改用非 loopback 位址，仍須依後文設定至少 24 字元的 `PLATEAI_API_TOKEN`。

Manifest 範例在 `configs/service/v3-local.example.json`，保存本次指定模型 hash。將你有權使用的對應資產放入指定目錄，或另存一份本機設定並改絕對路徑；相對路徑以設定檔所在目錄為準。換成其他可信 checkpoint 須重新核對來源與 hash，而不是刪掉檢查。用 `-Config 'D:\private\plate-service.json'` 指定。範例不含模型，直接啟動且無資產時應維持 not-ready。

- `cu118`：GTX 1080；以 `requirements/service-cu118-observed.txt` 約束本機實測版本。
- `cu128`：RTX 5060 / 5090 的 PyTorch profile；**Paddle 先裝 CPU 版**，直到在對應機器驗證 Paddle GPU 相容性。回應會如實顯示混合裝置。
- `cpu`：PyTorch CPU + Paddle CPU；此乾淨安裝尚未在無 GPU 機器實測，不能以本次強制 CPU 測試代替。
- cuDNN 編譯／載入 patch 版本仍有警告，本輪跑通不等於已清除部署風險。先 import torch 後 import paddle 的順序已固定。

上述新機 bootstrap 與 50 系列 profile 尚未實機重建驗收；本次 HTTP 使用已預檢的隔離環境。

## 介面

- `GET http://127.0.0.1:1788/api/health`：只表示程序存活。
- `GET http://127.0.0.1:1788/api/ready`：載入、hash、暖機完成才 200；包含實際 GPU／CPU、fallback 原因、初始化次數、runtime 版本。載入中／故障回 503。
- `POST http://127.0.0.1:1788/api/predict`：單圖多牌。
- `GET http://127.0.0.1:1788/docs`：Swagger；前端文件資產由 Swagger 公共 CDN 載入，推論不依賴它。

檔案：

```powershell
curl.exe -sS -F 'file=@D:\photos\car.jpg' http://127.0.0.1:1788/api/predict
```

Base64（也接受 `data:image/jpeg;base64,...`、PNG／WebP 對應 MIME）：

```powershell
$payload = @{ image_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes('D:\photos\car.jpg')) } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:1788/api/predict' -ContentType 'application/json' -Body $payload
```

兩者使用相同解碼／推論；不接受遠端 URL、本機檔案路徑或 ZIP。使用檔案內容判斷 JPEG／PNG／WebP，拒絕動畫；data URI 宣告的 MIME 必須與內容一致。

結果包含 `schema_version`、`request_id`、`image.width/height`、`model_id`、`detections`、`timings_ms`。每牌含 `id`、`bbox`、`crop_bbox`、`text`、`raw_text`、`display_text`、`status`、`reason`、`format_check`、`warnings`。

- 座標基於 **EXIF 方向校正後完整影像**；按 bbox 左上 y、x 排序編號。`bbox` 未加 buffer；`crop_bbox` 為實際裁切 `[x1,y1,x2,y2]` 整數座標，右下 exclusive。
- 不回傳圖片或 OCR 信心百分比。找到牌但讀不出仍保留框。
- `recognized` 只代表提取到單一、符合保守英數形狀的 OCR 文字，**不是法規驗證**。第一個 API 切片尚未載入完整官方編碼表：`format_check.status=unknown`、`allocation_status=unknown`、`registration_status=not_checked`；警告請呼叫方複核。
- 不做 O/0、I/1、B/8 替換；前導零、中文與未知符號不刪除。軍牌、古董牌與其他特殊配號沒有冒充驗證過。英文 OCR 可能漏中文字頭，故每筆均帶能力限制提醒。
- 非一般文字形狀回 `unverified_format`；多個合理牌號列回 `ambiguous`／text null；空白回 `unreadable`；單個裁切資料錯誤回 `error`。
- 頂層 `ok`／`partial`／`no_plate` 都是 HTTP 200。`timings_ms.total` 包含伺服器讀取請求到組回應之前時間，不等於客戶端完整 HTTP 延遲。

## 保護與錯誤

12 MiB 圖片、20 MP、multipart body 13 MiB、JSON body 17 MiB、base64 本體 16 MiB；每圖最多 20 個候選，超過回 422，不截斷冒充全結果。所有解析與圖片資料只在記憶體，不建立上傳暫存檔。

錯誤格式：`{"request_id":"...","error":{"code":"...","message":"..."}}`。

| HTTP | 常見 code |
| --- | --- |
| 400 | invalid_json / invalid_base64 / invalid_image / invalid_multipart |
| 401 | unauthorized |
| 408 | upload_timeout |
| 413 | image_too_large |
| 415 | unsupported_image / unsupported_media_type |
| 422 | missing_image / ambiguous_image_input / too_many_candidates |
| 429 | busy（Retry-After: 1） |
| 503 | model_not_ready / inference_failed |
| 504 | inference_timeout |

同時只處理一個請求，不排無界佇列。10 秒推論 timeout 或客戶端取消不代表 native 工作已取消，直到工作真的結束前維持 busy。致命 runtime 錯誤撤下 ready，需操作者檢查並重啟。GPU→CPU fallback 只在啟動暖機時進行，不逐圖重建。

預設 loopback，CORS 關閉。要開內網必須自行設定至少 24 字元隨機 token：

```powershell
$env:PLATEAI_API_TOKEN = '<由你的秘密管理工具提供的隨機值，勿提交 Git>'
.\run_api_1788.ps1 -BindAddress '0.0.0.0'
```

predict、ready、docs、openapi 需要 `Authorization: Bearer ...`；health 公開存活狀態。跨網路應使用既有 HTTPS 反向代理，不直接裸露至 Internet。不要把 token 塞 query string；有 token 的 Swagger 頁面也需由可送 Authorization header 的 client／受保護 proxy 開啟。

伺服器不落盤圖片、裁切、牌號，不開 Uvicorn access log；只記 request_id、結果狀態、候選數、總耗時、model_id。驗證腳本屬**明確 opt-in 的本機報告工具**，會把結果存到指定 output，與服務預設資料保留不同。

## 本次證據與限制

實測報告位於本機 `runs/plate-service/http-gpu-v1.json`。16 張舊完整場景回歸、三種輸入一致、200 次單牌／三牌交錯請求，模型初始化一次。

| 同機 HTTP（100 次／固定案例） | P50 | P95 |
| --- | --- | --- |
| 單牌 clean-000002 | 88.8 ms | 129.7 ms |
| 三牌 multi-000006 | 153.9 ms | 180.4 ms |

這不是獨立盲測或 50 張品質放行，也不是十張不同單牌／四牌各百次驗收。真正無牌負例、特殊車牌、長期記憶體趨勢、跨主機 HTTPS、乾淨 CPU-only／RTX 50、斷網啟動均仍待驗證；白圖只能算 no-plate 功能 smoke。1688 整合與正式 release 包也未完成。

另在隱藏 GPU、強制 CPU 模式對同一 16 張圖片跑通檔案／Base64／data URI、no-plate smoke 與常駐重用，詳 `runs/plate-service/http-cpu-functional-v1.json`。此輪同時有其他回歸工作，不用其兩筆延遲樣本宣稱 CPU P95。

可重跑（指定新 output，已有報告不覆寫）：

```powershell
& $env:PLATEAI_SERVICE_PYTHON .\tools\validate_plate_service.py --manifest 'D:\mytools\3waPlateAI-experiments\plate-service-phasea-20260925\manifest.json' --output '.\runs\plate-service\http-rerun.json'
```
