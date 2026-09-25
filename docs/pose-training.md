# Pose 車牌定位：準備、訓練與搬移

在 1688「模型訓練」頁使用 **YOLO Pose 車牌定位**。此工具是本機研究工作流，不是公開訓練服務；Studio 沒有多使用者權限隔離，請限制在受信任本機／內網，不要公開裸露 1688。

## 新機準備（Windows / CPython 3.11）

先建立 Studio 主環境（`run_build.ps1 -BootstrapOnly`），再另建 Pose 環境。以下腳本使用 uv，不會改主 `.venv` 或 API `.venv-service`：

```powershell
.\setup_pose.ps1 -Profile cu118
```

GTX 1080 用 `cu118`；RTX 5060／5090 用 `cu128`；無 GPU 用 `cpu`。固定 PyTorch 2.7.1、torchvision 0.22.1、Ultralytics 8.3.203。既有 `.venv-pose` 不覆寫；安裝失敗時保留現場供排查，不會自動刪除重建。此 bootstrap 的 NumPy 1.26／OpenCV 4.10 和早期共享主環境的實驗不同，不承諾 bit-identical 重現。RTX 50、乾淨 CPU-only 主機仍需實機驗收。

接著自行確認來源和使用權，明確 opt-in 下載。沒有圖片，只拿修正點不能訓練：

```powershell
.\.venv-pose\Scripts\python.exe tools\prepare_pose_inputs.py --download-pretrained --download-dataset --acknowledge-unreviewed-license
```

- 初始 `yolo11n-pose.pt` 取自 Ultralytics 官方 assets，鎖定 SHA-256。**它是初始權重，不是已訓練的車牌 detector。**
- 圖片取自 pinned `EZCon/taiwan-license-plate-detection` revision `ab64ba1e86615c8371e1b5617792a130d45028e8`；下載 parquet 快取後只還原固定的 2,000 train 與 119 validation-source 圖片。不是拿最新版本重抽。
- `annotations/pose-pilot-inputs-v1.json` 只含 train 的 source index／圖片 hash／`original_corners_sha256` 原始角點指紋，不含原始點座標；`pose-clean119-v1.json` 含人工修正點和 83／36 分組。還原與訓練都驗 hash。
- 資料在忽略的 `datasets/restricted/pose-ezcon-cache`、`out/ezcon-detector-v1`，初始模型在 `models/private/yolo11n-pose.pt`。既有資料目錄不覆寫；若不完整，先自行搬到備份位置，再還原，勿直接刪除人工修正 workspace。
- 下載是明確的本機實驗行為；來源資料、Ultralytics 和衍生模型之商用／再散布權利尚未完成審查，不由本 repo 的 MIT 授權涵蓋。外部來源移除時下載會失敗，不改用別的 detector。

也可用 `PLATEAI_POSE_PYTHON` 指定可信的獨立 Python。舊工作機仍相容 `runs/yolo-pose-spike-20260925/env/Scripts/python.exe` 和同目錄的初始權重，不強迫重裝；下載資料另需 pyarrow。此環境只給背景訓練 worker，不影響 1788。

## 網頁操作

1. 開啟 Studio 1688 → 模型訓練 →「檢查訓練條件」。列出 Python/CUDA、119 標註、初始模型及圖片缺項，附補齊指令；預檢不下載。
2. 勾選本機實驗確認、選裝置、按「開始 Pose 訓練」。與 CTC 共用單一背景任務鎖，不同時搶訓練；重整頁面會接回進度。想停止請按「中止」，關瀏覽器不會停止。
3. 逐檔檢查 bytes／decoded-pixel 重複、hash、方向、幾何；原始 train 排除 3 張退化標註，實際 1,997＋83＝2,080 張。36 張 holdout 不進訓練或挑模型。
4. 固定 10 epochs、640px、batch 8、FP32、seed 42、AdamW/cosine、無隨機幾何／HSV augmentation。取固定最後一輪 `last.pt`，然後單獨評估 36 張；不取 `best.pt`。原生訓練器可能在最後一輪驗證 train split，那不是 holdout 指標。
5. 完成後下載 ZIP：`pose-pilot.pt`、`detector-manifest.json`、`training-summary.json`、`README.txt`。不含圖、標註或 OCR，不自動啟用，原始 runs 與 log 保留在本機。

預檢只看可用性；完整圖片 hash／像素檢查在 worker 執行。GPU auto 只依 CUDA 可用性選裝置，實際 CUDA/OOM 訓練錯誤會記錄並停止，不悄悄切 CPU 重跑。每個 batch 檢查中止，啟動／原生儲存等步驟需等安全邊界。

## 模型搬到 3wa / 1788

這個包只解決 detector 缺件，**不是完整 API bundle**。確認使用權與品質後，把 `.pt` 放到目標私有模型目錄；另外準備官方 PaddleOCR 模型。在該主機自己的 service manifest 中設定 detector 路徑和本次新 hash、OCR 路徑和 hash，重啟後檢查 `/api/ready` 再傳圖驗收。`detector-manifest.json` 是搬移指紋，不可直接當完整 service config。

ZIP 僅供私有搬移。原生 Ultralytics checkpoint 可能保留本機訓練路徑等 metadata，不是匿名化公開發行包；不要公開上傳。公開的 119 點座標 JSON 則使用欄位白名單，不含本機路徑。

不能把新檔案硬套舊範例的 hash，也不要關掉 hash 檢查。重訓的新模型不自動沿用舊模型的速度／辨識成果。

36 張已被多次研究檢視，是內部 holdout，不是全新獨立盲測；也不是 OCR 文字正確率。評估沿用 640px 空間的 bbox AP50、四角全數 ≤8px 的 recall/precision、平均 corner error 及候選數。近重複／同場景家族仍非完整稽核。
