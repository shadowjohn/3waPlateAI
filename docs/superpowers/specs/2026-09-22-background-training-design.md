# Web Studio 單機背景訓練規格

日期：2026-09-22。狀態：使用者已核准並指示進入 plan；尚未實作。對應[實作計畫](../plans/2026-09-22-background-training.md)。

## 1. 使用情境與範圍

訓練是低頻操作：先訓練出穩定模型，平常持續使用該模型；效果不足或出現新式車牌時，再人工啟動調整。目標是在目前 Windows 單機 Web Studio 上可靠地完成一筆訓練，允許關閉網頁、關閉及重啟 Web server，再接回同一筆任務。

採用「每筆任務一個獨立 worker、SQLite autocommit 記錄、日誌檔、本機檔案鎖」。沿用既有資料集、CTC 訓練及 ONNX 匯出能力。這是小範圍的程序生命週期調整，不建立常駐排程服務。

本版不做任務排隊、自動重試、定期重訓、多 GPU 調度、跨機器協作、Redis/Celery、Windows Service、自動開機啟動、斷電續訓或模型管理平台。下載、合成、benchmark、release 等其他 Web 任務維持現有流程；本規格的持久化保證只涵蓋訓練。

## 2. 現況依據

- `tools/run_server.py` 與 `src/plateai_web/cli.py` 都使用 `reload=True`；`src/plateai_web/tasks.py` 的任務與子程序追蹤存在 Web 程序記憶體。Web 重啟後不能憑原 ID 恢復追蹤。
- `src/plateai_web/trainer.py` 在 Web 的背景 thread 內啟動 CLI 並讀取 stdout 管線；`training/engine.py` 完成整個 epoch 及驗證後才輸出指標，第一輪容易長時間看似沒有進展。
- 訓練引擎使用暫存目錄，成功後才發布正式 run 目錄。因此「正式 run 目錄尚不存在」不能單獨證明訓練未啟動或失敗。
- Web 包裝層在沒有指標時會補入 accuracy `98.5`、loss `0.05`，也會以 train loss 推算缺少的 val loss；這些值不能作為訓練結果。
- `trainer.py` 引用了未匯入的 `TaskStatus`；停止按鈕目前在請求成功後就宣稱已停止。改接 worker 時一併修正這兩個訓練流程問題。

404 只表示當下服務找不到 task ID；不把它當作 CUDA 失敗或 trainer 死亡的直接證據。

## 3. 程序與資料流

1. Web 收到開始要求，檢查參數，新增一筆 `pending` 紀錄，再啟動隱藏的獨立 Python worker，回傳原有 `task_id` 格式。
2. worker 取得本機訓練鎖，以單一條件 UPDATE 確認仍為可接手的 pending 任務、登記程序身分並改為 `running`；成功後才載入 PyTorch 等重量級模組，依序準備資料、訓練、驗證、匯出。若已收到取消要求則直接確認取消，不進入訓練。
3. worker 直接呼叫既有 `generate_dataset`（需要準備資料時）、`train_recognizer`、`export_crop_bundle` 的 Python 介面。訓練運算留在持有鎖的同一程序，避免 worker 結束後仍遺留未追蹤的訓練子程序。
4. Web 輪詢 SQLite 與任務日誌顯示進度。Web 結束不停止 worker；重開後透過既有 API 接回。
5. worker 完成、失敗或確認取消後寫入終態，釋放鎖並結束。沒有工作時沒有常駐 worker。

worker 的 stdin 接空裝置，stdout/stderr 直接接任務日誌檔，採 UTF-8 與即時 flush，不依賴 Web 持有的 PIPE。Windows 啟動方式必須在實機驗證能承受 Web 的 Ctrl+C、關閉啟動視窗及正常重啟，且不彈出額外主控台。程式更新後，已在跑的 worker 繼續本次工作；需要套用新訓練程式時，由使用者停止後重新開始。

## 4. SQLite：autocommit，不使用顯式交易

使用 Python 標準庫 `sqlite3`，配合目前 Python 3.11，以 `isolation_level=None` 開啟連線。

- 不使用 `BEGIN`、`COMMIT`、`ROLLBACK`、交易 context manager 或跨多個 SQL 的應用程式交易；不把整段訓練包在資料庫鎖內。
- 每次操作是獨立、參數化的短 SQL。需要一起變動的任務欄位放在同一個 `UPDATE`；各程序／thread 使用自己的連線，讀取後立即完成 cursor。
- SQLite 僅記錄任務；GPU 單筆執行限制由下一節的 OS 檔案鎖保證。採預設 journal 模式，不啟用 WAL 或額外資料庫服務，也不關閉 journal／durability。
- 設定有限的 busy timeout（5 秒）。失敗時回報明確錯誤，不無限重試；無法建立任務紀錄時不得啟動 worker。worker 無法持續寫入狀態時，記錄至日誌並在下一安全邊界停止，不宣告成功。

「不使用交易模式」在此指不由應用程式管理交易。SQLite 每一筆 SQL 仍有內部的隱式短交易，這是 autocommit 的正常行為，不能宣稱完全沒有交易。[Python 3.11 交易控制](https://docs.python.org/3.11/library/sqlite3.html#transaction-control)、[SQLite 隱式交易說明](https://www.sqlite.org/lang_transaction.html)。

只設一張 `training_tasks` 表，資料分為：

| 資料 | 內容 |
| --- | --- |
| 任務識別與輸入 | ID、名稱、建立時間、輸入參數 JSON，包含 epochs、資料集與輸出名稱 |
| 顯示狀態 | status、phase、progress、message、更新時間 |
| 執行與取消 | worker PID、程序啟動身分、心跳時間、最後進度時間、cancel_requested |
| 結果 | 指標／輸出路徑 JSON、錯誤原因；未知值使用 null |

Web 只新增任務及設定取消旗標；worker 更新進度與結果，不覆寫取消旗標。終態寫入使用帶原狀態條件的單一 UPDATE，避免舊心跳或停止請求把已完成任務改回執行中。Web 僅在確認 worker 已不存在時修正遺留狀態。

## 5. 單筆訓練與異常處理

同一個 checkout 的 worker 使用同一個 OS 排他檔案鎖，從準備資料開始持有到匯出及結果寫入結束。worker 結束後由 OS 釋放；鎖檔仍存在不表示鎖仍被持有，不能用「檔案是否存在」判斷忙碌。

開始 API 看見既有 pending/running 任務時，維持 `409` 與既有 task ID 回應。極短時間內兩個要求都通過預查時，只有取得 OS 鎖的 worker 可以進入訓練；另一筆標記 `failed`、原因 `training_busy`，不排隊。這個邊界接受「請求已受理但隨後顯示忙碌」，不承諾兩個 API 要求一起進行資料庫交易。

狀態沿用 `pending → running → completed / failed / cancelled`；未進入 running 的啟動失敗／取消可由 pending 直接進入終態。取消要求不立即改成 cancelled，先維持原非終態並顯示「停止中」。

- 啟動失敗：保留任務並標記 failed，顯示原因。
- 啟動途中 Web 結束：重新連線時核對 worker。超過 30 秒仍無 worker 接手的 pending 任務標記啟動中斷；遲到的 worker 必須先確認仍可接手，不能復活終態。
- worker 消失／電腦重開：核對 PID 與程序啟動身分，確認原 worker 不存在後標記 failed，原因為訓練中斷；不自動重新訓練。
- 心跳超過 15 秒未更新但 worker 仍存在，或無權確認程序身分：顯示「暫時無回應／狀態待確認」，保留停止操作，不僅憑心跳逾時判定已死，也不解除訓練互斥。

worker 每約 3 秒回報存活，並獨立記錄最後一次實際進度；Web 讀取或心跳不能偽造 batch 已前進。程序檢查使用唯讀方式，不向可能被重用的 PID 發送終止訊號。

## 6. 真實進度與停止

階段為 `preparing`、`training`、`validating`、`exporting`。訓練引擎增加可選的進度 callback 與取消檢查，未提供時維持現有 CLI 行為；Web 指標來自結構化事件與正式 report，不再依靠正則解析人類閱讀的 log。

- 每個 batch 邊界可檢查取消；對外進度限流到約每 1 秒一次，epoch 與階段切換立即更新。
- 顯示 epoch、batch/total、實際 train loss、最後進度時間。validation 完成前，val loss/accuracy 顯示「尚未驗證」，不補猜測值。
- epoch 曲線只使用完整 epoch 的真實指標；第一個 epoch 尚未完成時，明確提示並展示 batch 進度。
- 整體百分比保留準備／訓練／匯出的階段權重，標示為工作進度而非剩餘時間預估；100% 只在成功驗證輸出後出現。
- 停止 API 只寫入取消要求；worker 在資料生成 callback、train/validation batch 邊界及匯出前後檢查。原生匯出呼叫進行中時顯示等待安全停止，不承諾立即打斷。
- worker 清理自己尚未發布的暫存輸出並確認不再訓練後，才寫入 cancelled。Web 必須持續輪詢到終態，收到停止 API 的 200 不等於已停止。
- 本版不新增強制終止按鈕。重複停止或對終態停止不修改既有結果。取消與完成競爭時，以 worker 的終態寫入為準；不能將已完成任務重新標為 cancelled。

## 7. 檔案與模型使用

任務資料放在已被 Git 忽略的 `runs/.web-training/`：`tasks.sqlite3`、`training.lock` 與 `logs/<task_id>.log`。資料庫／日誌不放進某筆正式訓練輸出目錄，維持引擎「暫存後發布、既有目錄拒絕覆寫」契約。

訓練成果仍輸出到 `runs/<run_name>/`；每次 ONNX 匯出至 `models/bundles/train-<task_id>/`。輸出名稱須為目錄名且解析後位於指定根目錄下；不覆寫已有 run、bundle 或合成資料。訓練需要自動生成資料時，沿用既有挑選／seed 規則並呼叫底層生成介面，不借用會把整筆任務標記 completed 的 Web 合成包裝層。

worker 只有在 checkpoint、正式 report、匯出 bundle 與既有 parity／bundle 驗證成功後才寫入 completed。匯出失敗時任務為 failed，但已成功發布的 checkpoint/report 保留；失敗原因和已產生路徑一併顯示。沒有真實指標就顯示缺失，不填入固定準確率。

已在使用的 `active-v1` 模型保持原樣；本版完成訊息明確區分「已訓練／匯出」與「已啟用」。使用者確認新模型後再人工決定替換，本版不新增自動啟用、版本切換 UI 或模型發布流程。已發布產物不因取消要求或清理動作刪除。

## 8. Web 與啟動入口相容性

- `run_server.bat`、`tools/run_server.py`、`plateai_web.cli` 預設停用 reload，batch wrapper 轉傳參數；`--dev-reload` 明確啟用開發模式。訓練 worker 仍獨立於 reload 程序。
- 保留 `/api/train/start`、`/api/train/active`、`/api/train/stop`、`/api/tasks/{id}` 與既有主要回傳欄位。訓練查詢接到 SQLite，其他任務沿用記憶體管理器；新的訓練 ID 仍為 12 位 hex，建立時避免與現有任務 ID 衝突。
- `/api/train/active` 納入 pending/running，沒有活動任務時可回傳最近一次訓練結果。重新開頁／Web 重啟後接回同一 ID、歷史指標及最新日誌。
- stop 增加明確的 `cancel_requested` 與 `task_id`；保留既有回應鍵時，`cancelled` 僅能表示已確認停止，不能以請求受理冒充。頁面與 API 同步調整。
- API 暫時連不上時顯示重新連線中；對真正未知的 ID 才回 404。舊版記憶體任務在升級前已遺失的資料不能還原，UI 遇到這類 404 應停止無限輪詢並提示重新查詢。
- 訓練標題使用實際的 PyTorch CTC 名稱。GPU 記憶體圖保留為資源資訊，不作為訓練成功或正在前進的判據。

## 9. 實作涉及的範圍

新增薄型 SQLite task store 與 worker 入口，集中放在 `src/plateai_web/`；調整訓練包裝層、訓練引擎的 callback／取消邊界、既有 API、啟動入口及訓練頁。保留其他 TaskManager 功能、既有訓練演算法、資料契約與輸出驗證。使用 Python 標準庫提供 SQLite、程序與 Windows 檔案鎖能力；不為本版新增外部服務或通用工作排程框架。

## 10. 驗收條件

1. 啟動短訓練後，關頁再開可看到同一 ID、epoch/batch 與日誌；第一個 epoch 完成前也有真實進度。
2. Windows 實機以 Ctrl+C 停止 Web、關閉啟動視窗、再重新啟動；worker 持續計算，原 ID 仍可查，重新連線後可以停止。另驗證 `--dev-reload` 重載不影響同筆訓練。
3. 同時送兩個開始要求，實際最多一個 worker 進入準備／訓練，其餘得到 409 或 `training_busy` 終態。
4. 停止請求不立即冒充 cancelled；在準備、train、validation、匯出階段，以及重複停止、取消與完成競爭時，狀態與產物保留符合本規格。
5. 使用自建測試程序模擬 worker 異常退出及 Web 在啟動途中退出；任務仍可查且能辨識中斷，不復活終態、不把無關程序當成原 worker。
6. SQLite 使用 autocommit；以 SQL trace 確認應用程式沒有送出顯式交易指令。新連線可立即讀到更新，Web 輪詢／取消不因整段訓練持有資料庫鎖而阻塞。
7. 完成任務須檢查真實 checkpoint/report/bundle；缺少指標、訓練失敗、匯出失敗各有真實回報，沒有 `98.5`／`0.05` 或推算 val loss 的假資料。
8. 重複訓練使用不同輸出，不覆寫已使用模型；其他 Web 任務與既有 CLI 測試不退化。任務紀錄、日誌與模型仍留在 Git 忽略範圍。

目前證據限於現有程式閱讀、Python/SQLite 官方語義核對及文件自檢。以上是未來實作的驗收要求，不代表測試已通過；尚未修改或重啟 server、建立資料庫、啟動訓練、驗證瀏覽器或驗收 GPU。遠端部署、IIS、正式資料庫與 production 不在本版驗收範圍。
