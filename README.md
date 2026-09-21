# 3waPlateAI

3waPlateAI 是一套用於台灣車牌風格合成、訓練契約與高速辨識流程的 MIT 授權工具組。

本公開儲存庫刻意維持為**純原始碼**：提供程式、設定、JSON Schema、文件與四張無害的合成 CI fixture，但不發布 checkpoint、訓練權重、ONNX 模型、TensorRT engine 或 Model Bundle。資料權利、法規審查、訓練算力、模型調校與後續維護均由使用者負責。

M1 提供可重現的 CPU 車牌裁切合成；M2 提供本機 PyTorch CTC 訓練與 ONNX 匯出；M3a 提供確定性的四角點校正；M3b 提供本機原生姿態偵測工作流；M4 提供本機完整 Bundle Reader 核心。效能基準、服務化與正式部署仍屬後續工作；詳細設計請見[設計規格](docs/superpowers/specs/2026-09-21-3wa-plate-ai-design.md)。

## 目前內容

| 區域 | 職責 | 狀態 |
| --- | --- | --- |
| `plateai_shared` | 訓練與推論共用的不可變規則、字元集與 JSON 契約 | 可用 |
| `plateai_trainer.synthetic` | 規則抽樣、乾淨渲染、確定性增強與交易式資料輸出 | 可用 |
| `plateai_reader` | 已驗證的本機 ONNX session、偵測後處理、校正、批次辨識與受規則限制的 CTC 解碼 | M4 Reader 核心可用，未附模型 Bundle |
| Model Bundle | 模型、字元集、規則、張量、批次、解碼與校正的雜湊驗證契約 | Schema 可用，未發布 Bundle |

## 快速開始

開發與 CI 支援 CPython 3.11。

### Windows 一鍵建置

在 PowerShell 執行：

```powershell
.\build.ps1
```

`build.ps1` 會建立被忽略的 CPython 3.11 `.venv`（優先使用 `uv`，否則使用 `py -3.11`）、安裝鎖定的訓練與測試依賴、執行完整測試、建立 `dist/`、強制安裝剛建立的 wheel、驗證 `plateai-read --help`、執行三張圖的已安裝套件合成 smoke，最後執行 `pip check`。`build.bat` 是可由 cmd 或雙擊呼叫的包裝器。

常用選項包括 `-BootstrapOnly`（只建立或更新環境）、`-SkipTests`、`-SkipPackage`，以及 `.venv` 不符合 CPython 3.11 時的 `-RecreateVenv`。後者只會移除被忽略的 `.venv`，不會碰觸原始碼、資料集、模型或其他輸出路徑。建置只產生 Python 套件產物，不會訓練、下載或發布模型。

### 手動建立環境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/py311.lock.txt
python -m pip install --no-deps -e .
plateai-generate generate --count 100 --seed 42 --output out\demo
python -m pytest
```

生成器拒絕覆寫既有輸出目錄。輸出採交易式發布：失敗時只清除自己建立的暫存目錄，不會移除不相干檔案。

### PyTorch 與 GPU 顯卡版本支援

若要在本機使用 GPU 加速訓練（例如 `plateai-train --device cuda`）：
- **舊架構顯卡 (如 GTX 1080 / Pascal)**：PyTorch 請選用 **`cu118`** (CUDA 11.8) 建置版本（例如 `torch==2.5.1+cu118`）。若使用較新的 CUDA 版本可能缺少 sm_61 算力支援而無法啟動。
- **新架構顯卡 (如 RTX 5060、RTX 5090 / Blackwell)**：請選用 **`cu128`** (CUDA 12.8+) 或支援 Blackwell 新架構的最新 PyTorch 建置版本。
- **CPU 開發環境**：官方依賴鎖定檔 (`requirements/py311.training.lock.txt`) 採用純 CPU 依賴 (`torch==2.14.0`)，適合無獨立顯卡或 CI 環境運行全流程測試與確定性合成。

## M1 合成車牌資料

- 預設輸出 RGB `uint8` 的 380×160 PNG，對應新式自用小客車比例，並記錄來源或變換後的四角點。
- OCR canonical label 不含裝飾用連字號；顯示字串保留連字號。
- 每一樣本由 run seed 與 index 推導獨立 seed；在鎖定的 Python 3.11 依賴下，字串、增強參數、metadata 與影像皆可重現。
- 預設新式自用小客車為白底黑字、`LLL-DDDD`，並排除 `I`、`O`、`4`。
- 預設字型是以 OFL-1.1 授權隨附的 Noto Sans Mono 視覺近似字型。若要用本專案的台灣車牌專用字型生成訓練資料，請明確指定 `--font taiwan_plate`：

  ```powershell
  .\.venv\Scripts\plateai-generate generate `
    --count 10000 `
    --seed 42 `
    --font taiwan_plate `
    --output out\train-taiwan-plate-font
  ```

  每次生成會記錄字型檔名與 SHA-256。`TaiwanPlate-Regular.ttf` 是以專案內的公路局參考資料建立，供本機合成與視覺比較使用；它不表示主管機關以產品形式發布此字型，或已授予衍生材料的一般再散布權。

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
  涵蓋全台車牌文字格式（新式七碼 `LLL-DDDD`、舊式六碼 `LL-DDDD` / `DDDD-LL`、機車六碼 `LLL-DDD`、機車混合碼 `LLD-DDD` / `DLL-DDD`，以及舊式五碼/四碼 `LL-DDD` / `DDD-LL` / `LL-DD` / `DD-LL`），並補齊包含數字 `4` 的 34 字元集（10 數字 + 24 字母，排除 `I`、`O`）。
  > **注意**：`tw_standard_v1.json` 是「統一 OCR 字串規則」（plate_type 均為 standard，供模型學習全台合法號牌文字結構），不是用於辨識車牌底色或車種的視覺模板；若需要特定底色（如黃牌、紅牌、綠牌）之視覺渲染，請搭配色牌模板。

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

選用的 `training` extra 提供符合裁切契約的本機 PyTorch CTC trainer 與 ONNX exporter。它使用 Pillow golden 灰階前處理、80 個 CTC timestep、blank index 0，支援動態類別數（預設 34 類，統一規格為 35 類），並在發布被忽略的本機 Bundle 前檢查 PyTorch 與 ONNX parity。資料集、權重、ONNX、run 與 Bundle 均不提交；詳見[訓練與資料集指南](docs/training.md)。

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

專案原始碼採 [MIT License](LICENSE)。MIT 不會重新授權相依套件、外部字型、資料集或模型；請見[第三方告知](THIRD_PARTY_NOTICES.md)。
