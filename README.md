# 3waPlateAI

<p align="center">
  <img src="assets/readme_banner.jpg" alt="3waPlateAI Banner" width="100%">
</p>

3waPlateAI 是台灣車牌辨識與驗證工具組，現在以「**傳入完整照片，回傳每張車牌的文字、位置與狀態 JSON**」為主要接入方式。
提供獨立的 **1788 車牌 API**，以及由「3wa 老司機看板娘」領航的 **1688 Web Studio**；原有合成資料、訓練、ONNX Reader 與四角幾何研究工具仍保留。

目前開發主線為 `main`。截至 2026-09-25，獨立 API 已完成本機 GPU／強制 CPU 功能驗證；**可供本機專案接入測試，但尚未完成正式部署、品質與模型授權放行**。1688 尚未串接這套新 API，兩個入口的辨識管線不可混為一談。

本公開儲存庫刻意維持為**純原始碼**：提供程式、設定、JSON Schema、文件與四張無害的合成 CI fixture，但不發布 checkpoint、訓練權重、ONNX 模型、TensorRT engine 或 Model Bundle。資料權利、法規審查、訓練算力、模型調校與後續維護均由使用者負責。

目前服務採用：完整場景圖 → YOLO bbox → 每側 5% buffer → PP-OCRv3 mobile／英文辨識 → 多牌 JSON。使用原生 PyTorch／Paddle，模型啟動時載入、暖機並常駐重用；不要求呼叫方先裁切或人工點角。

這條 API 路徑只使用既有 Pose checkpoint 的 **bbox**，不使用四角點，也尚未加入旋轉／透視拉正、二值化或 ONNX。這些不是第一版接入的前置條件；後續依實際失敗案例再改善。

文件入口：[API 接入文件](docs/plate-service-api.md) · [Pose 懶人訓練／補齊 detector](docs/pose-training.md) · [最小服務規格與後續範圍](docs/superpowers/specs/2026-09-25-minimal-plate-recognition-service-design.md) · [訓練指南](docs/training.md) · [開發紀錄](history.md) · [參考資料來源](reference.md)。

## 目前內容

| 區域 | 職責 | 狀態 |
| --- | --- | --- |
| `plateai_service`／1788 | 常駐 YOLO＋PaddleOCR、檔案／Base64、多牌 JSON、GPU／CPU 與容量保護 | 本機 HTTP 已驗證；正式發行未完成 |
| `plateai_web`／1688 | 合成、訓練監控、即時辨識、候選 A/B、Benchmark 圖文對照 | 既有 Studio 可用；尚未代理新 1788 API |
| `plateai_shared` | 訓練與推論共用的不可變規則、字元集與 JSON 契約 | 可用 |
| `plateai_trainer.synthetic` | 規則抽樣、乾淨渲染、確定性增強與交易式資料輸出 | 可用 |
| `plateai_reader` | 已驗證的本機 ONNX session、偵測後處理、校正、批次辨識與受規則限制的 CTC 解碼 | M4 Reader 核心可用，未附模型 Bundle |
| Model Bundle | 模型、字元集、規則、張量、批次、解碼與校正的雜湊驗證契約 | Schema 可用，未發布 Bundle |

M1–M4 是合成、訓練、四角校正與 Reader 的既有工作流，詳見[原始設計規格](docs/superpowers/specs/2026-09-21-3wa-plate-ai-design.md)。新 API 不需先跑這套訓練／匯出流程，也不會替換 Studio 現役 Bundle。`release/3wa_plate_api` 仍是舊 source-only 雛形，**不是新 API 的啟動入口**。

## 快速開始

開發與 CI 支援 CPython 3.11。

### 獨立車牌 API（1788，本機測試版）

在專案根目錄執行。API 使用獨立 `.venv-service`，不要把 Paddle／NumPy 相依套件裝入主訓練 `.venv`；不需要先跑 `run_build`、`run_train` 或開著 Studio。

1. 先安裝 `uv`，再建立服務環境；已有 `.venv-service` 時腳本會拒絕覆寫。

   ```powershell
   .\setup_api_1788.ps1 -Profile cu118
   ```

2. 自行備妥可信且有權使用的模型。儲存庫**不附模型，也不會在推論時自動下載**；範例設定 [v3-local.example.json](configs/service/v3-local.example.json) 指向 `models/private/pose-pilot.pt`、`models/private/PP-OCRv3_mobile_det/`、`models/private/en_PP-OCRv3_mobile_rec/`，並鎖定對應 SHA-256。只有相符資產才能通過；使用其他可信模型時須另建設定並核對來源與 hash。

3. 已放妥與範例相符的資產後啟動，或用 `-Config` 指定自己的 manifest。相對模型路徑以設定檔所在目錄為準。

   ```powershell
   .\run_api_1788.ps1 -Config .\configs\service\v3-local.example.json
   ```

已配置本機 `runs/plate-service/config.json` 的環境可直接執行根目錄 `run_api_1788.bat`。既有隔離 Python 可用 `-Python 'D:\private\plate-env\Scripts\python.exe'` 或 `PLATEAI_SERVICE_PYTHON` 指定；詳細步驟見 [API 接入文件](docs/plate-service-api.md)。

| 服務環境 profile | 裝置與目前驗證範圍 |
| --- | --- |
| `cu118` | GTX 1080：既有 Windows 隔離環境已實測 PyTorch 2.7.1+cu118、Paddle GPU 3.2.2、PaddleOCR 3.2.0 |
| `cu128` | RTX 5060／5090：PyTorch cu128；Paddle 暫用 CPU 版，尚未在對應硬體驗收 |
| `cpu` | PyTorch／Paddle CPU；已做既有 GPU 主機的強制 CPU 功能測試，未完成乾淨 CPU-only 主機驗收 |

新機 bootstrap 尚未實機重建驗收，不能把上述既有環境證據當成所有 profile 都已通過。`-Device auto` 預設在啟動／暖機時逐元件嘗試 GPU，失敗才退回 CPU；可用 `-Device cpu` 強制 CPU。實際裝置以 `/api/ready` 為準。

#### 傳圖取 JSON

開啟 [Swagger 試用頁](http://127.0.0.1:1788/docs)。先確認 [就緒狀態](http://127.0.0.1:1788/api/ready) 回 HTTP 200；`/api/health` 只表示程序存活，不代表模型已載入。

檔案上傳：

```powershell
curl.exe -sS -F 'file=@D:\photos\car.jpg' http://127.0.0.1:1788/api/predict
```

Base64（同樣接受 `data:image/jpeg;base64,...`）：

```powershell
$payload = @{ image_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes('D:\photos\car.jpg')) } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:1788/api/predict' -ContentType 'application/json' -Body $payload
```

回應節錄（示意值；完整欄位見 API 文件）：

```json
{
  "schema_version": 1,
  "status": "ok",
  "image": { "width": 640, "height": 480 },
  "detections": [
    {
      "id": 1,
      "bbox": [120.5, 180.25, 320.5, 260.25],
      "crop_bbox": [110, 176, 331, 265],
      "text": "ABC1234",
      "raw_text": "ABC-1234",
      "display_text": "ABC-1234",
      "status": "recognized"
    }
  ]
}
```

`detections` 為多牌陣列；讀不出文字仍保留框與失敗狀態，沒有偵測到車牌則回 `status: "no_plate"` 與空陣列。座標是 EXIF 方向校正後完整影像的 `[x1,y1,x2,y2]`：`bbox` 未加 buffer，`crop_bbox` 是實際裁切整數框，右下邊界不包含在裁切內。完整回應另含 `request_id`、`model_id`、`timings_ms`、每牌 `reason`、`format_check` 與 `warnings`；不回 OCR 信心百分比。

`recognized` 只代表 OCR 字串提取成功且符合保守英數形狀，**不代表已核發、合法或真牌**。官方完整編碼／配號查核尚未接入；軍牌、古董牌與中文特殊字頭仍須複核，不會用 O/0、I/1 等猜字修正冒充辨識成功。

#### 容量與部署邊界

- 支援 JPEG／PNG／WebP 靜態圖片，每次一張、最多 12 MiB／20 MP／20 個候選；不接受 URL、伺服器檔案路徑或 ZIP。
- 單 worker 常駐處理，同時只接一件工作，忙碌回 `429`；載入中或故障回 `503`，推論逾時回 `504`。請求取消／逾時不會提前釋放仍在運算的 native 工作。
- 「熱載」指載入、暖機後重用模型，不是不停機換權重；換模型須更新可信 manifest 並重啟。
- 預設只綁 `127.0.0.1`。開內網須設定至少 24 字元隨機 `PLATEAI_API_TOKEN`，跨網路使用 HTTPS 反向代理；不要直接公開裸露服務。
- 服務不落盤上傳圖、裁切或車牌文字；本機驗證報告工具則是明確 opt-in。完整錯誤碼與 token 用法見 [API 接入文件](docs/plate-service-api.md)。

#### 已量測的本機結果（2026-09-25）

Windows／GTX 1080，detector 與 OCR 都使用 GPU；模型暖機後，單牌／三牌固定案例各 100 次交錯呼叫，200 次請求共用一次初始化。以下是**客戶端 HTTP 延遲**，不含服務冷啟動：

| 固定場景 | P50 | P95 |
| --- | --- | --- |
| 單牌 | 88.8 ms | 129.7 ms |
| 三牌 | 153.9 ms | 180.4 ms |

16 張既有完整場景的 API 文字集合與 Phase A v3 結果一致，檔案／Base64／data URI 輸入一致；強制 CPU 亦完成同批功能回歸。這些是**既有案例回歸，不是獨立盲測準確率或正式效能承諾**。報告保留於本機忽略的 `runs/plate-service/`，不隨 Git 發布。

下一步是 1688／3wa 專案整合與完整場景驗收；PHP `api.php?mode=tw_plate_ocrv&version=1` 仍是後續構想，尚未實作。真正無牌負例、特殊牌、長時間記憶體、斷網啟動、乾淨新機與 RTX 50 環境仍待驗收，Paddle cuDNN patch 版本警告也尚未排除。

### 缺少 pose-pilot.pt？從 Studio 訓練

1688「模型訓練」頁新增 **YOLO Pose .pt 懶人訓練**：檢查缺件 → 背景訓練／中止 → 下載 detector、hash、設定與評估 ZIP。不會替換現役模型，也不是 OCR 訓練。

先用 `setup_pose.ps1 -Profile cu118` 建獨立環境（RTX 50 用 `cu128`），依 [Pose 準備說明](docs/pose-training.md) 取得 pinned 初始權重和圖片，再按網頁按鈕。原始 1,997＋人工修正 83 張訓練，36 張保留最後評估，固定 10 epochs 取 `last.pt`。

提供 [119 張修正點](annotations/pose-clean119-v1.json) 和 hash 對照；**不提供圖片、原始標註或訓練後模型**。新機需自行取得相符圖片；只有點不能訓練。新模型有新 hash，搬移到 1788 時須另行配置 OCR／service manifest 並重新驗證，資料與模型授權邊界不變。

### Windows 一鍵建置

在 PowerShell 執行：

```powershell
.\run_build.ps1
```

`run_build.ps1`（亦可雙擊或執行 `run_build.bat`）會建立被忽略的 CPython 3.11 `.venv`（優先使用 `uv`，否則使用 `py -3.11`）、安裝鎖定的訓練與測試依賴、執行完整測試、建立 `dist/`、強制安裝剛建立的 wheel、驗證 `plateai-read --help`、執行三張圖的已安裝套件合成 smoke，最後執行 `pip check`。

常用選項包括 `-BootstrapOnly`（只建立或更新環境）、`-SkipTests`、`-SkipPackage`，以及 `.venv` 不符合 CPython 3.11 時的 `-RecreateVenv`。建置只產生 Python 套件產物，不會訓練、下載或發布模型。

### 訓練／建置環境的 PyTorch profile

一鍵建置腳本 `run_build.bat` / `run_build.ps1` 支援 `-Cuda` 參數（預設 `auto`，依本機顯卡選擇）。**這是主專案 `.venv` 的建置設定，不是上方 API 的 `.venv-service`**；目前腳本／lockfile 指定：

- **GTX 1080／Pascal**：`-Cuda cu118`，`torch==2.7.1+cu118`。
- **RTX 5060／5090／Blackwell**：`-Cuda cu128`，`torch==2.11.0+cu128`。
- **CPU／CI**：`-Cuda cpu`，訓練 lockfile 指定 `torch==2.14.0`。

版本列出的是儲存庫設定，不代表每種硬體／全新安裝都已驗收；不要把訓練環境版本直接套到 Paddle 服務環境。

範例：

```powershell
.\run_build.bat -Cuda cu128       # 為 RTX 5060 / 5090 一鍵安裝與建置
.\run_build.bat -Cuda cu118       # 為 GTX 1080 一鍵安裝與建置
.\run_build.bat -BootstrapOnly    # 自動偵測本機顯卡，僅初始化環境不跑測試
```

### 一鍵訓練與 ONNX 匯出 (`run_train.bat` / `run_train.ps1`)

提供開發者最直覺的一鍵訓練指令：

- **自動偵測 GPU**：預設優先使用 CUDA 訓練，無顯卡時自動退回 CPU。
- **自動補齊訓練資料**：若尚未合成訓練集，自動以專用車牌字型（`taiwan_plate`）與台灣標準規則快速合成 2,000 張訓練樣本與 500 張驗證樣本。
- **自動匯出 ONNX Bundle**：訓練完成並通過 Native/ONNX parity 驗證後，匯出至本機指定的 bundle 位置；啟用到既有辨識流程必須另行人工確認。

```powershell
.\run_train.bat                     # 一鍵自動化訓練與 ONNX 導出 (預設 10 epochs, auto-gpu)
.\run_train.bat -Epochs 20          # 指定訓練 20 輪
.\run_train.bat -Device cpu         # 強制使用 CPU 訓練
```

### 一鍵啟動 Web Studio (`run_server.bat`)

啟動 [Web Studio](http://127.0.0.1:1688/)（Port 1688），提供合成資料產生器、模型訓練監控、即時辨識與 Benchmark：

```powershell
.\run_server.bat
```

Benchmark 報告可逐張對照圖片、GT、辨識文字及文字版 bbox；文字不符標紅，圖片使用原生 lazy loading，並區分 GT 裁切與模型定位。這是既有 Studio 管線的報告，不是新 1788 API 的結果頁；四角人工審核／Pose 比較工具仍是獨立研究流程，尚未整合到 Studio。

### Web Studio 背景訓練

Web Studio 預設以不啟用 reload 的方式啟動；開發時才明確加上旗標：

```powershell
.\run_server.bat
.\run_server.bat --dev-reload
```

從訓練頁開始的工作會由獨立背景 worker 執行，所以關閉 Web Studio 或開發 reload 不會自動停止它；請回到訓練頁按「中斷訓練」，等待畫面確認已停止。每次 Web 訓練會建立新的 `runs\<run-name>` 與 `models\bundles\train-<task-id>`，完成只代表「已匯出，尚未啟用」。請先檢視 bundle 與驗證結果，再依你的部署流程人工選擇是否啟用模型。

### 一鍵下載真實測試照片 (`run_get_test_data.bat` / `run_get_test_data.ps1`)

自動從公開真實台灣車牌資料集（EZCon）下載實際道路車輛與車牌照片至 `test_data/`，供本機測試檢驗：

- 影像自動命名為 `{序號}_{真實車牌號碼}.jpg`（例如 `002_MYX-6873.jpg`），一眼即可辨識 Ground Truth。
- 自動生成 `labels.json` 記錄真實標註與旋轉框座標。
- 使用多執行緒下載；數量與速度依上游資料及網路狀態而定。使用前仍須確認[資料來源與權利邊界](docs/evaluation-sources.md)。

```powershell
.\run_get_test_data.bat               # 下載預設 30 張台灣真實車牌測試照片至 test_data\
.\run_get_test_data.bat -Count 50     # 下載 50 張
.\run_get_test_data.bat -Count all    # 下載上游目前提供的全部 test split 照片
```

## M1 合成車牌資料

- 預設輸出 RGB `uint8` 的 380×160 PNG，對應新式自用小客車比例，並記錄來源或變換後的四角點。
- OCR canonical label 不含裝飾用連字號；顯示字串保留連字號。
- 每一樣本由 run seed 與 index 推導獨立 seed；在鎖定的 Python 3.11 依賴下，字串、增強參數、metadata 與影像皆可重現。
- 預設新式自用小客車為白底黑字、`LLL-DDDD`，並排除 `I`、`O`、`4`。
- 預設字型是以 OFL-1.1 授權隨附的 Noto Sans Mono 視覺近似字型。若要用本機建立的台灣車牌字型生成訓練資料，先執行下列兩個批次檔；產物只會寫入已忽略的 `assets/local/fonts/`，不會進入 Git 或 release package：

  ```powershell
  .\build_prepare_ttf_env.bat
  .\build_taiwanplate_ttf.bat
  ```

  再明確指定 `--font taiwan_plate`：

  ```powershell
  .\.venv\Scripts\plateai-generate generate `
    --count 10000 `
    --seed 42 `
    --font taiwan_plate `
    --output out\train-taiwan-plate-font
  ```

  每次生成會記錄字型檔名與 SHA-256。`TaiwanPlate-Regular.ttf` 是以專案內的公路局參考資料建立，僅供本機合成與視覺比較；它不表示主管機關以產品形式發布此字型，或已授予衍生材料的一般再散布權。

- 機車色牌採 `plate_type` 選擇模板，不從任意車牌字串猜測車種。可產生一般重機白底黑字、250–550cc 黃底黑字、550cc 以上紅底白字，以及 50cc 綠底白字：

  ```powershell
  .\.venv\Scripts\plateai-generate generate `
    --charset configs\charsets\tw_new_style_private_passenger_v1.txt `
    --rules configs\plate_rules\tw_motorcycle_colours_v1.json `
    --template configs\plate_templates\tw_motorcycle_colours_v1.json `
    --output out\motorcycle-colours
  ```

  此組輸出尺寸與類型可變，供視覺與後續多 profile 工作使用；不可直接餵給目前固定 380×160 小客車契約的 M2/M4 recognizer。

- **全台統一 OCR 字串規則（Unified Standard Profile）**：
  收錄常見車牌文字形狀（新式七碼 `LLL-DDDD`、舊式六碼 `LL-DDDD` / `DDDD-LL`、機車六碼 `LLL-DDD`、機車混合碼 `LLD-DDD` / `DLL-DDD`，以及舊式五碼/四碼 `LL-DDD` / `DDD-LL` / `LL-DD` / `DD-LL`），並補齊包含數字 `4` 的 34 字元集（10 數字 + 24 字母，排除 `I`、`O`）。
  > **注意**：`tw_standard_v1.json` 是訓練／合成使用的「統一 OCR 字串規則」（plate_type 均為 standard），不是完整現行法規、配號或車籍資料庫，也不是用於辨識車牌底色或車種的視覺模板；若需要特定底色（如黃牌、紅牌、綠牌）之視覺渲染，請搭配色牌模板。

  合成資料生成：
  ```powershell
  .\.venv\Scripts\plateai-generate generate `
    --charset configs\charsets\tw_standard_v1.txt `
    --rules configs\plate_rules\tw_standard_v1.json `
    --font taiwan_plate `
    --count 10000 `
    --seed 42 `
    --output out\train-std
  ```

  訓練 35 類（34 字元 + Blank）辨識模型與導出 ONNX Bundle：
  ```powershell
  .\.venv\Scripts\plateai-train --train out\train-std --validation out\val-std --output runs\std-cpu --charset configs\charsets\tw_standard_v1.txt --rules configs\plate_rules\tw_standard_v1.json --epochs 10
  .\.venv\Scripts\plateai-export --checkpoint runs\std-cpu\best.pt --report runs\std-cpu\report.json --output models\bundles\tw-std-v1 --charset configs\charsets\tw_standard_v1.txt --rules configs\plate_rules\tw_standard_v1.json
  ```

`tests/fixtures/synthetic/` 下的四張 PNG 是演算法建立的玩具輸入，只用於驗證程式與契約，不含真實車輛或可識別的車牌資料，也不構成辨識準確度聲明。

## M2 本機辨識訓練

選用的 `training` extra 提供符合裁切契約的本機 PyTorch CTC trainer 與 ONNX exporter。它使用 Pillow golden 灰階前處理、80 個 CTC timestep、blank index 0，支援動態類別數（預設 34 類，統一規格為 35 類），並在發布被忽略的本機 Bundle 前檢查 PyTorch 與 ONNX parity。

- **自動硬體偵測（預設 `--device auto`）**：執行訓練時自動偵測 GPU，預設優先使用 CUDA 進行 GPU 加速訓練；若系統未偵測到可用顯卡或 CUDA 不可用，則自動平順退回 CPU 訓練。使用者亦可明確指定 `--device cuda` 或 `--device cpu`。
- 資料集、權重、ONNX、run 與 Bundle 均不提交；詳見[訓練與資料集指南](docs/training.md)。

## M3a 四角點校正

`plateai_reader.rectifier` 接受 `uint8` RGB 來源影像及任意排列的四個有限、在畫面內的角點。它先檢查凸包、拒絕退化或方向不明的幾何，再以車牌長邊判定上下與左右語意，輸出給 M2 使用的固定 RGB `160×380` 裁切圖。

Warp 目標是離散像素 `(0,0)`、`(379,0)`、`(379,159)`、`(0,159)`，使用 OpenCV 雙線性插值與白色常數邊界。identity regression 會檢查所有外框輸出像素都對應到來源座標，避免最後一列或最後一欄混入邊界色。M3a 不含偵測資料、偵測訓練、偵測權重或偵測 ONNX；那些是 M3b 的範圍。

## M3b 本機姿態偵測

M3b 是給使用者自行提供、且具合法使用權背景圖 manifest 的本機工作流。原生偵測器輸入為 640×640 OpenCV RGB letterbox，輸出 pre-NMS `[batch,8400,13]` candidates。確定性 NMS 後，每個保留的四角點會獨立交給 M3a 做固定 RGB `380×160` 校正；單一壞姿態不會阻止其他候選進入 recognizer 邊界。

資料合成、偵測器訓練與完整 Bundle 匯出，請依[偵測器工作流](docs/training.md#local-m3b-detector-workflow)執行。原始碼驗收刻意不包含背景圖、偵測權重、ONNX、TensorRT benchmark、瀏覽器整合或正式環境辨識指標。

## M4 本機 Reader

`plateai-read` 只會在驗證 manifest、宣告檔案雜湊與 ONNX 契約後載入使用者自行訓練的本機完整 Bundle。它保留 detector 與 recognizer 的 ONNX Runtime session，預設依序偏好 TensorRT、CUDA、CPU，執行確定性 NMS 與校正，依 recognizer manifest 分批處理標準化裁切圖，並用受規則限制的 CTC 解碼。例如：

```powershell
.\.venv\Scripts\plateai-read --bundle models\bundles\v1-full-local --image C:\local\frame.png --warmup 5
```

命令輸出 JSON，包含保留的偵測結果、個別拒絕原因、canonical/display 文字、偵測信心值與各階段觀察時間。時間僅是本機量測，不是 GPU 效能或實地準確度聲明。儲存庫仍不含完整 Bundle、ONNX、真實影像或 benchmark 結果。

## 用自己的資料訓練

使用公開生成器，僅匯入具有合法權利的資料集，並針對自己的領域訓練 Bundle。真實車牌照片、正式資料、沒有再散布權的字型及訓練產物不應放入此儲存庫。可用的外部評估來源與權利邊界請見[評估來源說明](docs/evaluation-sources.md)；本機模型政策見[models/README.md](models/README.md)。

## 授權

專案原始碼採 [MIT License](LICENSE)。MIT 不會重新授權相依套件、外部字型、資料集或模型；尤其新 API 使用的 Ultralytics 與研究 checkpoint，仍須分別確認適用授權及資料衍生權重用途，不能因本 repo 採 MIT 就推定可直接商用／再散布。請見[第三方告知](THIRD_PARTY_NOTICES.md)與[模型政策](models/README.md)。
