# Web Studio Background Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓低頻、單機單筆訓練在背景完成，Web 關閉或重啟後仍能查詢同一任務、看到真實進度及要求停止。

**Architecture:** 每筆訓練啟動一個獨立 Python worker，直接執行資料準備、CTC 訓練與 ONNX 匯出，持有 OS 檔案鎖直到結束。SQLite autocommit 記錄任務，stdout/stderr 寫入日誌；Web 只提交要求、查詢及設定取消旗標。沿用既有演算法和輸出驗證，所有新成果另存。

**Tech Stack:** Windows、CPython 3.11、標準庫 sqlite3/subprocess/msvcrt/ctypes/threading、既有 FastAPI/PyTorch/ONNX Runtime/jQuery/ECharts、pytest。

**Spec:** [已核准的背景訓練規格](../specs/2026-09-22-background-training-design.md)。使用者於 2026-09-22 回覆「好，進 plan」；本文件只規劃實作，不代表已執行。

## Global Constraints

- 「本版不做任務排隊、自動重試、定期重訓、多 GPU 調度、跨機器協作、Redis/Celery、Windows Service、自動開機啟動、斷電續訓或模型管理平台。」
- 「使用 Python 標準庫 `sqlite3`，配合目前 Python 3.11，以 `isolation_level=None` 開啟連線。」
- 「不使用 `BEGIN`、`COMMIT`、`ROLLBACK`、交易 context manager 或跨多個 SQL 的應用程式交易；不把整段訓練包在資料庫鎖內。」
- 「採預設 journal 模式，不啟用 WAL 或額外資料庫服務，也不關閉 journal／durability。」busy timeout 為 5 秒。
- 「worker 每約 3 秒回報存活，並獨立記錄最後一次實際進度」；15 秒無心跳只表示待確認，pending 30 秒無人接手才核對啟動中斷。
- 「對外進度限流到約每 1 秒一次，epoch 與階段切換立即更新。」只有真正完成才是 100%。
- 「本版不新增強制終止按鈕。」取消請求受理與確認取消是不同狀態。
- 任務資料為 `runs/.web-training/{tasks.sqlite3,training.lock,logs/<task_id>.log}`；正式訓練為 `runs/<run_name>/`；匯出為 `models/bundles/train-<task_id>/`。
- 「已在使用的 `active-v1` 模型保持原樣」；其他 Web 任務沿用現有 TaskManager，持久化保證只涵蓋訓練。
- 保留既有編碼/BOM/CRLF、CLI 呼叫相容性和 public API 主要欄位；依賴不增加。模型、資料及任務紀錄不得進 Git。
- 執行前確認工作樹，保留 spec/history 等既有修改。每項完成檢查自己的 diff；未取得提交指示前不自動 commit/push。

## Review Focus

1. Web 在 INSERT 後或啟動 worker 途中離開、取消與 claim 同時發生：遺留 pending 可收斂，遲到 worker 不復活終態；Tasks 1、4。
2. Windows venv 啟動器 PID 與實際 Python PID 不同、PID 重用或查詢被拒絕：由 worker 登記自己的身分，不能誤判或誤殺程序；Task 4。
3. 日誌有中文/空白路徑/未完整行，SQLite 忙碌或不可寫：回傳有界日誌與明確錯誤，不掉成假成功或靜默失去追蹤；Tasks 1、4、5。
4. 既有 validation 目錄 seed 相同、charset/rules 不符、輸出重名：拒絕錯誤輸入或建立獨立新資料，保留舊資料；Task 3。
5. 首輪尚無 validation、取消碰上匯出完成、使用者正在用舊模型：不補假數值、不刪已發布產物、不自動啟用新模型；Tasks 2、3、4、5。

## File Structure and Order

| Task | 檔案 | 責任 |
| --- | --- | --- |
| 1 | 新增 `src/plateai_web/training_store.py` | 單表 autocommit CRUD、條件式狀態更新、日誌讀取 |
| 2 | 新增 `src/plateai_trainer/training/control.py`；修改 `training/engine.py` | 輕量事件型別、batch 進度及合作式取消 |
| 3 | 修改 `src/plateai_web/trainer.py` | 從 subprocess 包裝改成可直接呼叫的訓練／匯出流程 |
| 4 | 新增 `src/plateai_web/training_process.py`、`training_worker.py` | Windows 程序身分、檔案鎖、背景啟動、心跳及恢復判斷 |
| 5 | 修改 `src/plateai_web/app.py`、`cli.py`、`tools/run_server.py`、`run_server.bat`、`web/js/app.js`、`web/index.html` | API/啟動入口/訓練頁接線，其他頁面保留 |
| 6 | 修改 `README.md`、`docs/training.md`、`history.md`；新增 `tests/integration/test_background_training_flow.py` | 真實短訓練與 Web 重啟整合驗收、操作文件 |

測試檔及支援檔列於各 Task。順序為 1 → 2 → 3 → 4 → 5 → 6；全部在本任務依序執行即可，建議 Native，介面緊密相依，不需要為每項重開工作環境。

### Task 1: SQLite autocommit 任務記錄

**Files:** Create `src/plateai_web/training_store.py`, `tests/unit/test_training_store.py`.

**Interfaces:** 所有方法在同一檔案，回傳一般 dict，不另外建立 ORM 或 repository 層。

- `TrainingStore(state_dir: Path, *, clock: Callable[[], float] = time.time, trace: Callable[[str], None] | None = None)`；公開 `state_dir: Path`，constructor 只保存設定，不立即開啟 DB。
- `create(task_id: str, name: str, request: dict) -> dict`、`get(task_id: str) -> dict | None`、`active() -> list[dict]`、`latest() -> dict | None`。
- `claim(task_id: str, pid: int, start_token: str) -> bool`：只接手 pending 且未要求取消的任務。
- `snapshot(task_id: str, *, phase: str, progress: int, message: str, result: dict, advanced: bool) -> bool`；只更新 running，`advanced` 才改最後進度時間。
- `heartbeat(task_id: str) -> bool`、`request_cancel(task_id: str) -> bool`、`cancel_requested(task_id: str) -> bool`。
- `finish(task_id: str, status: str, *, message: str, error: str | None = None, result: dict | None = None) -> bool`：status 只接受 completed/failed/cancelled；不可覆寫既有終態。completed 只允許從 running 且未有取消旗標轉入；failed/cancelled 可從 pending 或 running 轉入。
- `fail_if_unchanged(observed: dict, error: str) -> bool`：以讀取時的 status、updated_at、worker_pid/start_token 為條件，供 Task 4 收斂遺留紀錄。
- `log_path(task_id: str) -> Path`、`read_logs(task_id: str, limit: int = 100) -> tuple[list[str], int]`：最後完整行與總完整行數。僅接受 12 位小寫 hex ID。

- [ ] **1.1 寫入可觀察行為測試。** 主測試如下；另涵蓋重複 claim、終態不可覆寫、心跳不更新 last_progress_at、取消與進度各自欄位不互相覆蓋、壞 ID 不組成路徑、SQLite 錯誤向上回報、中文日誌只返回最後 100 完整行。

```python
from plateai_web.training_store import TrainingStore

def test_autocommit_is_visible_and_cancel_blocks_claim(tmp_path):
    sql = []
    directory = tmp_path / "任務 記錄"
    writer = TrainingStore(directory, trace=sql.append)
    reader = TrainingStore(directory)
    task_id = "0123456789ab"
    writer.create(task_id, "模型訓練", {"epochs": 1})
    assert reader.get(task_id)["status"] == "pending"
    assert writer.request_cancel(task_id)
    assert not reader.claim(task_id, 123, "process-start-1")
    assert reader.get(task_id)["cancel_requested"] is True
    assert not any(s.lstrip().split()[0].upper() in
                   {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT"} for s in sql)
```

- [ ] **1.2 執行 RED。** `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_store.py -q`；預期模組不存在而失敗，保存結果。
- [ ] **1.3 建立單表和短 SQL 實作。** 欄位為 id/name/request_json/status/phase/progress/message/result_json/error/worker_pid/worker_start_token/cancel_requested/created_at/updated_at/heartbeat_at/last_progress_at。id 是 primary key；JSON 以 `allow_nan=False` 序列化，讀取時解析成 request/result。每次 `_connect()` 設定 row_factory/trace，使用 `contextlib.closing` 關閉連線，不能以 `with connection:` 管理交易。首次資料操作才建立目錄/DB；Schema 只用 `CREATE TABLE IF NOT EXISTS`，不導入 migration 框架。

```python
with closing(self._connect()) as connection:
    now = self.clock()
    changed = connection.execute(
        "UPDATE training_tasks SET status='running', worker_pid=?, "
        "worker_start_token=?, updated_at=?, heartbeat_at=? "
        "WHERE id=? AND status='pending' AND cancel_requested=0",
        (pid, start_token, now, now, task_id),
    ).rowcount
return changed == 1
```

`_connect()` 使用 `sqlite3.connect(db_path, isolation_level=None, timeout=5.0)`。動態 UPDATE 欄位須固定白名單，值全部 bind；單句更新 result 與 progress。request_cancel 只改旗標與更新時間；finish 不清除已存 result。cancel_requested 查不到自己任務時應報錯並停止，不當作 false。read_logs 以 streaming + deque 限制記憶體，忽略最後未換行的半行；不存在的 log 回傳 `([], 0)`。

- [ ] **1.4 執行 GREEN 與 diff 檢查。** 重跑 1.2；加入真實兩連線測試，確認無額外 commit 就能看見更新。用可注入 clock，避免以長時間 sleep 測狀態條件。

### Task 2: 真實訓練事件與安全取消邊界

**Files:** Create `src/plateai_trainer/training/control.py`, `tests/unit/test_training_progress.py`; modify `src/plateai_trainer/training/engine.py`. Reuse `tests/conftest.py` 的 `v1_train_dir` / `v1_validation_dir`。

**Interfaces:** control 模組不 import torch 或 Web。

```python
from dataclasses import dataclass

class TrainingCancelled(RuntimeError):
    """The user requested cancellation at a safe boundary."""

@dataclass(frozen=True, slots=True)
class TrainingProgress:
    kind: str
    phase: str
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    val_acc: float | None = None
```

- kind 為 stage/batch/epoch；phase 為 preparing/training/validating/exporting；val_acc 在引擎內為 0–1 比例，到 Web 相容格式才乘以 100。
- `train_recognizer(config: TrainingConfig, *, on_progress: Callable[[TrainingProgress], None] | None = None, check_cancelled: Callable[[], None] | None = None) -> TrainingRun`。
- `evaluate_recognizer(model, loader, codec, device, *, on_batch: Callable[[int, int], None] | None = None, check_cancelled: Callable[[], None] | None = None) -> EvaluationReport`；既有四個 positional 參數不變。

- [ ] **2.1 寫 RED 測試，確認首輪 batch 先於 epoch 指標出現，而且沒有提前填 val 值。** 再測取消發生於第一個 train batch、第一個 validation batch，以及發布 run 前；輸出與自己 staging 都未殘留。取消清理沿用現有 engine 的 `except BaseException` 路徑。

```python
from plateai_trainer.training.engine import TrainingConfig, train_recognizer

def test_first_batch_precedes_epoch_metrics(v1_train_dir, v1_validation_dir, tmp_path):
    events = []
    run = train_recognizer(
        TrainingConfig(v1_train_dir, v1_validation_dir, tmp_path / "run",
                       epochs=1, batch_size=2, device="cpu"),
        on_progress=events.append,
    )
    first = next(e for e in events if e.kind == "batch" and e.phase == "training")
    completed = next(e for e in events if e.kind == "epoch")
    assert events.index(first) < events.index(completed)
    assert (first.batch, first.total_batches) == (1, 2)
    assert first.train_loss is not None and first.val_acc is None
    assert completed.val_acc == run.report["history"][0]["val_acc"]
```

- [ ] **2.2 執行 RED。** `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_progress.py -q`；預期缺少 callback 參數／事件。
- [ ] **2.3 把 epoch list comprehension 改成明確 batch 迴圈。** 在資料載入前後、每個 train/validation batch 前後、匯出正式 run 前呼叫可選取消函數；callback 異常不能被吞掉。最小迴圈形式如下，`on_progress` 省略時維持原數學與 RNG 順序。

```python
losses = []
for batch_index, batch in enumerate(train_loader, 1):
    if check_cancelled is not None:
        check_cancelled()
    losses.append(train_one_batch(model, batch, optimizer).loss)
    if on_progress is not None:
        on_progress(TrainingProgress(
            kind="batch", phase="training", epoch=epoch,
            total_epochs=config.epochs, batch=batch_index,
            total_batches=len(train_loader), train_loss=sum(losses) / len(losses),
        ))
    if check_cancelled is not None:
        check_cancelled()
```

validation 迴圈每個 batch 回報 `(index, len(loader))`，包含被規則拒絕而略過的 batch 邊界；epoch 事件在 history 加入後送出，採同一份四捨五入指標。先送 stage 事件再開始慢步驟。保留 CLI epoch 文字並加 `flush=True`；不改訓練超參數、最佳 checkpoint 選取或 report schema。

- [ ] **2.4 執行 GREEN 和既有 CLI 相容測試。** `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_progress.py tests/unit/test_training_model.py tests/unit/test_training_device.py tests/integration/test_training_cli.py -q`。callback/no-callback 在相同 seed 的小型 CPU run 應產生相同數值結果。

### Task 3: 同程序的資料準備、訓練與匯出

**Files:** Modify `src/plateai_web/trainer.py`; create `tests/unit/test_web_training_pipeline.py`. 底層 synthetic/export 模組沿用現有介面。

**Interfaces:** 保留 `find_available_datasets(root)`、`get_dataset_seed(path)`、`get_dataset_config_paths(path)`。新增 `validate_training_request(root: Path, task_id: str, request: dict) -> dict` 回傳正規化參數；`run_training_pipeline(root: Path, task_id: str, request: dict, *, on_progress: Callable[[TrainingProgress], None], on_result: Callable[[dict], None], check_cancelled: Callable[[], None]) -> dict` 回傳完整結果。

request 公開欄位仍為 epochs/run_name/train_dataset；內部測試可指定 validation_dataset、batch_size、device，Web 不因此增加控制項。預設 epochs 沿 API 的 5、batch_size 32、device auto、run_name `run-<task_id>`。model 輸出固定 `train-<task_id>`。root 指資料／產物根目錄；內建 configs/fonts 由既有 package/source resolver 取得，不假設測試 tmp_path 下也有一份原始碼。

- [ ] **3.1 寫 RED 測試。** 包含錯誤 run_name、已存在輸出不刪除、指定不存在訓練集不得靜默改選、同 seed 或不同 charset/rules 的 validation 不可沿用、驗證集生成的 callback 不能將主任務標成完成、匯出失敗保留 checkpoint/report、active-v1 sentinel hash 不變。

```python
import pytest
from plateai_web.trainer import validate_training_request

@pytest.mark.parametrize("name", ["..", "../other", "a/b", "a\\b", "C:\\outside", "CON"])
def test_run_name_cannot_escape_or_use_device_name(tmp_path, name):
    with pytest.raises(ValueError):
        validate_training_request(tmp_path, "0123456789ab", {"epochs": 1, "run_name": name})
```

- [ ] **3.2 執行 RED。** `./.venv/Scripts/python.exe -m pytest tests/unit/test_web_training_pipeline.py -q`；預期沒有新介面。
- [ ] **3.3 實作輸入與產物邊界。** task ID 必須合法；epochs 為真正 int 且 >=1；run_name 拒絕絕對路徑、分隔符、`.`/`..`、Windows 保留名稱、尾端空白/句點，resolve 後確認仍在 runs 下且不是 `.web-training`。已存在 run/bundle 直接拒絕。訓練資料只讀，使用既有 dataset 驗證。沿用現有資料挑選優先序，validation 需 seed 不同且 charset/rules 相符；既有預設 validation 若不合，生成 `val-<task_id>`，不刪舊目錄。沒有 train 時才生成原預設 2,000 張，validation 預設 500 張；config_paths 缺檔不靜默換成其他字集。
- [ ] **3.4 改以既有 Python 介面串接。** 模組頂端只留輕量 imports；函數內載入 `TrainingConfig/train_recognizer`、`GenerationRequest/generate_dataset`、`ExportRequest/export_crop_bundle`。生成所用 charset/rules 與選定訓練一致，使用現有生成 callback 檢查取消；前後也檢查，利用其 staging 清理。下列為真實訓練完成後的串接核心：

```python
run = train_recognizer(config, on_progress=on_progress, check_cancelled=check_cancelled)
on_result({"run_dir": str(run.output_directory),
           "checkpoint": str(run.best_checkpoint), "report": str(run.report_path)})
check_cancelled()
on_progress(TrainingProgress(kind="stage", phase="exporting"))
bundle = export_crop_bundle(ExportRequest(
    run.best_checkpoint, run.report_path, bundle_dir,
    charset_path=config.charset_path, rules_path=config.rules_path,
))
on_result({"bundle_dir": str(bundle)})
check_cancelled()
```

`config` 由上一步的選定 train/validation/config paths 建立；`bundle_dir = root / 'models' / 'bundles' / f'train-{task_id}'`。export_crop_bundle 已包含 parity 與 manifest 驗證，保留它的 no-replace 發布。pipeline 不持有 DB、不標記任務終態。on_result 是中途成果的 patch，讓匯出失敗／取消時仍留下已發布 run 的位置。

從 run.report 的 history/validation/train 取真實結果；對缺失指標保留 null，非有限數值報錯。Web `history/current_metrics/final_accuracy` 的 val_acc 只在這個相容邊界乘以 100； final_* 清楚表示最後一個 epoch，不能冒充已部署模型的實測成績。移除固定 `98.5`、`0.05`、`train_loss * 0.9`。新 pipeline 完成後回傳 `status: success`、完整歷史和產物路徑，但不啟用模型。

Task 5 切換 API 前保留既有 `train_model_task` 作暫時相容入口；切換完成且 `rg` 確認無 caller 後移除它，避免雙入口長期並存。

- [ ] **3.5 執行 GREEN。** 重跑 3.2，另跑 `tests/integration/test_export_bundle.py`。此 Task 用 fake exporter 測故障分支；真實 train→export 交給 Task 6。

### Task 4: 獨立 worker、OS 鎖與重啟後狀態判斷

**Files:** Create `src/plateai_web/training_process.py`, `src/plateai_web/training_worker.py`, `tests/unit/test_training_process.py`, `tests/unit/test_training_worker.py`, `tests/integration/test_training_worker_lifecycle.py`, `tests/helpers/background_training_probe.py`。

**Interfaces:**

- process 模組：`TrainingBusy(RuntimeError)`；`training_lock(path: Path) -> ContextManager[None]`；`current_process_identity() -> tuple[int, str]`；`process_state(pid: int, start_token: str) -> str` 回傳 alive/dead/unknown。
- `_spawn_detached(command: list[str], *, cwd: Path, log_path: Path) -> subprocess.Popen`；`launch_training_worker(root: Path, task_id: str) -> None`。
- `reconcile_training_tasks(store: TrainingStore) -> None`：只核對 active rows，呼叫 store.fail_if_unchanged 收斂確定失效的紀錄。
- worker 模組：`run_worker(root: Path, task_id: str, *, pipeline: Callable | None = None) -> int`；`main(argv: Sequence[str] | None = None) -> int` 使用 `--root`、`--task-id`。pipeline=None 時在 claim 後才匯入 Task 3 函數；注入 pipeline 僅供測試，不開放 HTTP 指定模組。

- [ ] **4.1 寫 RED 測試。** 真實兩個 OS process 爭鎖只允許一個；持鎖者正常／異常結束後可取得同一鎖檔。worker 接手前已取消則不呼叫 pipeline；fake pipeline 回報後取消，確認不是 completed。另測 heartbeat fault 使下一安全邊界停止、匯出結果已落地後取消仍保留路徑、stale heartbeat + alive/unknown 不判 dead、stale pending 不會被遲到 claim 復活。

```python
from plateai_web.training_store import TrainingStore
from plateai_web.training_worker import run_worker

def test_cancelled_before_claim_never_trains(tmp_path):
    store = TrainingStore(tmp_path / "runs" / ".web-training")
    task_id = "0123456789ab"
    store.create(task_id, "train", {"epochs": 1})
    store.request_cancel(task_id)
    def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("cancelled task reached training")
    assert run_worker(tmp_path, task_id, pipeline=forbidden_pipeline) == 0
    assert store.get(task_id)["status"] == "cancelled"
```

- [ ] **4.2 執行 RED。** `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_process.py tests/unit/test_training_worker.py tests/integration/test_training_worker_lifecycle.py -q`。
- [ ] **4.3 實作 Windows 檔案鎖及唯讀身分查詢。** Windows 以 `msvcrt.locking(fd, LK_NBLCK, 1)` 鎖 offset 0 的一個 byte，handle 在 context 內持續開啟；finally 解鎖/close，不刪 lock file。只將 lock contention 轉成 TrainingBusy，其他 I/O 錯誤保留原因。process_state 使用 ctypes 宣告完整 Win32 argtypes/restype，`OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE)`、`GetProcessTimes` 的建立時間 token、`WaitForSingleObject(handle, 0)` 判活性，finally CloseHandle。PID 不存在／token 不同為 dead；AccessDenied 或不明查詢錯誤為 unknown。不得使用 `os.kill(pid, 0)` 作 Windows 探測。Windows 為本版實機驗收平台；其他平台不支援時明確報錯且 Windows 測試標記 skip，不擴大跨平台保證。
- [ ] **4.4 實作背景啟動。** 以 `sys.executable -u -m plateai_web.training_worker --root <root> --task-id <id>` 啟動，cwd 用原始碼根目錄；從來源 checkout 執行時將其 src 加入 child 的 PYTHONPATH，避免讀到舊 wheel。複製既有環境，只覆加 Python UTF-8 設定，不讀取或輸出敏感環境值。

```python
with log_path.open("ab", buffering=0) as log_stream:
    child = subprocess.Popen(
        command, cwd=str(cwd), stdin=subprocess.DEVNULL,
        stdout=log_stream, stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True, env=child_environment,
    )
return child
```

`child_environment` 按上段規則建立；不用 shell、不加 CREATE_NEW_CONSOLE，也不與 DETACHED_PROCESS 疊加會被忽略的 CREATE_NO_WINDOW。Popen.pid 僅供 launcher 本身診斷，不當成訓練 worker 身分；worker 使用自身 os.getpid() 與建立時間登記。啟動失敗仍留 failed task/log。Web 不保存一條 stdout PIPE 或以退出 hook 終止訓練。
- [ ] **4.5 實作 worker 主流程及終態競爭。** 取得 OS 鎖、檢查任務/取消、條件 claim 成功後立刻啟動 3 秒 heartbeat thread，才 lazy import pipeline。主執行緒維護 result cache 和對外更新節流；heartbeat 只改 heartbeat_at，不改 batch/result。每個 check_cancelled 都確認取消與 heartbeat thread 記下的 storage error；storage error 應導向 failed，不能冒充使用者取消。

事件 stage 強制發布，batch 以 monotonic 每秒最多一次；epoch 寫 history。train 進度為 `10 + 75 * ((epoch-1 + 0.8*batch/total_batches) / total_epochs)`；validation 使用同 epoch 的最後 20%，完整 epoch 為 `10 + 75*epoch/total_epochs`；preparing 3、exporting 88。task.progress 最終才升 100。stage 事件沒有 batch 時不得除以零，且不將既有百分比倒退。event/result cache 合併使用一把程序內 threading.Lock，避免遺失中途成果；不要把這把鎖跨越 SQL 或訓練呼叫。

```python
try:
    result = pipeline(root, task_id, request, on_progress=on_progress,
                      on_result=on_result, check_cancelled=check_cancelled)
    on_result(result)
    check_cancelled()
    completed = store.finish(task_id, "completed", message="訓練與匯出完成，尚未啟用", result=result_cache)
    if not completed and store.cancel_requested(task_id):
        store.finish(task_id, "cancelled", message="已停止；已發布產物保留", result=result_cache)
except TrainingCancelled:
    store.finish(task_id, "cancelled", message="訓練已停止", result=result_cache)
```

此區塊的 request 來自 store.get、三個 callback 在 run_worker 內按以上規則定義；外層另捕捉一般異常，print traceback 並嘗試 finish failed。SQLite 本身不能寫入時只保留 stderr/log、exit 非零，交重連後 reconcile 判斷；不能無限補寫。finally 通知 heartbeat thread 結束並 join（受 5 秒 DB timeout 約束），再釋放 OS 鎖。completed/failed 的 exit code 為 0/1；正常 cancelled 為 0，實際 API 狀態仍是 cancelled。

reconcile 對超過 30 秒未 claim 的 pending 使用 snapshot 條件更新；running 必須 process_state=dead 才 fail。15 秒 heartbeat 過期但程序 alive/unknown 的提示由 API 附加 health 欄位，不能覆寫 worker status。遲到 worker claim 失敗直接離開。
- [ ] **4.6 執行 GREEN 與真實程序存活測試。** helper 的 parent 模式用同一 `_spawn_detached` 啟動 child 後自行結束；child 呼叫 `run_worker(root, task_id, pipeline=probe_pipeline)`，probe 使用與 Task 3 pipeline 相同簽名，每 0.1 秒送 batch 事件與 check_cancelled，最長 30 秒自行退出。測試確認 parent 結束後 heartbeat/batch 仍增加，再透過另一個 store 取消。另以 helper 的 crash 模式 `os._exit(9)` 測 OS 解鎖與失敗收斂；只控制測試建立的程序，不終止現有 server/trainer。等待用 deadline + 短輪詢，不使用固定長 sleep。重跑 4.2。

平台依據：[Python msvcrt.locking](https://docs.python.org/3.11/library/msvcrt.html#msvcrt.locking)、[Windows process creation flags](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags)、[GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)。flags 本身不能作為視窗關閉驗收證據；若執行環境額外使用會終止子程序的 Job Object，記錄實測限制，不自動安裝 service 或宣稱已達成。

### Task 5: API、啟動入口與訓練頁接線

**Files:** Modify `src/plateai_web/app.py`, `src/plateai_web/trainer.py`, `src/plateai_web/cli.py`, `tools/run_server.py`, `run_server.bat`, `web/js/app.js`, `web/index.html`, `tests/test_web_api.py`; create `tests/unit/test_training_api.py`, `tests/unit/test_web_launcher.py`。

**Interfaces:** `get_training_store() -> TrainingStore` 是 FastAPI dependency，lazy 建立 `ROOT/runs/.web-training`，不可在 import app 時建立 DB。訓練 endpoints 使用 `Depends(get_training_store)`，tests 以 dependency_overrides 換到 tmp_path。`cli.build_parser() -> argparse.ArgumentParser` 共用 `--dev-reload` 解析；`cli.main(argv=None)` 與 `tools/run_server.main(argv=None)` 都預設 reload=False。

- [ ] **5.1 先隔離現有 Web 測試。** `tests/test_web_api.py` 的 dataset 測試自行建立 tmp_path 範例並替換 app.ROOT；env/build 及 release/build endpoint 測試替換其 target 為只在新 TaskManager 留紀錄的 stub，不觸碰 checkout 的 release 或個人環境。release endpoint 測試改驗證 stub 被呼叫和任務結果，不再以 checkout 裡可能早已存在的 release 檔案作成功依據。每個測試注入自己的 TrainingStore、TaskManager 並在 teardown 清 dependency_overrides。保留 predictor 測試既有範圍，不修改其推論邏輯。
- [ ] **5.2 新增 RED 測試。** 覆蓋訓練 task 由新 store 重開後可查、pending 回 409、第二個並發要求至少受到 worker lock 保護、啟動錯誤留 failed row、stop 受理但尚未 cancelled、未知 task 404、訓練／非訓練 task 分流。兩條啟動入口都驗證 Uvicorn 收到 reload=False，指定 flag 才為 True，不真正啟動 server/browser。

```python
from starlette.testclient import TestClient
from plateai_web.app import app, get_training_store
from plateai_web.training_store import TrainingStore

def test_stop_only_requests_cancellation(tmp_path):
    store = TrainingStore(tmp_path / "state")
    task_id = "0123456789ab"
    store.create(task_id, "模型訓練", {"epochs": 1})
    app.dependency_overrides[get_training_store] = lambda: store
    try:
        with TestClient(app) as client:
            response = client.post("/api/train/stop")
            assert response.status_code == 200
            assert response.json()["cancel_requested"] is True
            assert response.json()["cancelled"] is False
            assert client.get(f"/api/tasks/{task_id}").json()["status"] == "pending"
    finally:
        app.dependency_overrides.clear()
```

- [ ] **5.3 執行 RED。** `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_api.py tests/unit/test_web_launcher.py -q`。
- [ ] **5.4 接入資料與程序介面。** start 先 reconcile/precheck；用 Task 1 create 的 PK 檢查與現有 TaskManager ID 查詢生成唯一 12 位 hex，再呼叫 Task 4 launcher。受理維持 200 + `{status:'started', task_id}`；已有任務為 409；輸入錯誤 422；DB 或 spawn 失敗回 503 並在能寫入時留下 failed ID。不能在 DB 建立失敗時仍 spawn。stop 只更新 pending/running 的旗標，回 `status/task_id/cancel_requested/cancelled/message`；無活動任務為 ok/false/false。既有 `/api/tasks/{id}` 先查記憶體任務，找不到再查訓練 DB，保持非訓練任務不受訓練 DB 故障影響；API 序列化統一既有 keys 加 phase/health/cancel_requested/heartbeat_at/last_progress_at。不要用失敗 DB 查詢假裝查無任務。

train/active 回傳活動任務或最近結果；訓練 task result 保留 history/current_metrics，缺失指標為 null，logs 最後 100 行。從 trainer 移除已無 caller 的舊 train_model_task、子程序正則解析和假數字。非訓練的 TaskManager.run_in_background/run_subprocess_command 不改。
- [ ] **5.5 修改 launcher。** 共用 parser 定義 `parser.add_argument('--dev-reload', action='store_true')`；傳入 `reload=args.dev_reload`。保留 port 1688、現有 host、UTF-8 與開瀏覽器習慣，`run_server.bat` 兩個 Python 呼叫都加 `%*`。主控台提醒「關閉工作台不停止已啟動訓練；請從訓練頁停止」。安裝依賴流程不擴寫。
- [ ] **5.6 修改訓練頁狀態回報。** 訓練使用獨立 timer/current task ID，避免既有全域 pollInterval 在點其他頁面時中斷訓練追蹤；不整頁重寫。一次最多一筆未完成查詢，過期 response 不覆蓋新 task。正常輪詢 1 秒，斷線後以 1/2/5/10 秒上限重連；404 停止該 ID 輪詢並查一次 active，不自動開始新訓練。

```javascript
const terminal = ["completed", "failed", "cancelled"].includes(task.status);
$("#btn-start-train").prop("disabled", !terminal);
$("#btn-stop-train").prop("disabled", terminal || task.cancel_requested);
const metrics = task.result.current_metrics || {};
$("#train-stat-vloss").text(metrics.val_loss == null ? "尚未驗證" : Number(metrics.val_loss).toFixed(4));
$("#train-stat-acc").text(metrics.val_acc == null ? "尚未驗證" : `${Number(metrics.val_acc).toFixed(1)}%`);
```

在 `updateTrainChartAndStats` 將統計卡更新移到 history 空檢查前；曲線僅用完整 epochs，首輪顯示 batch/total 與最後進度時間。stop 的成功 callback 只顯示「停止中」並繼續查詢，錯誤顯示停止要求未確認，不宣稱已清理。所有 server message/log/path 經 text() 呈現。完成時標示「已匯出，尚未啟用」和實際路徑；重開頁只恢復結果，不重複彈歷史成功 modal。將兩處 YOLO 訓練標題改為 PyTorch CTC。
- [ ] **5.7 執行 GREEN。** 重跑 5.3 與隔離後的 `tests/test_web_api.py`；如果本機有 node，執行 `node --check web/js/app.js`。頁面實際互動放到 Task 6 驗收，不用 grep 命中字串冒充瀏覽器測試。

### Task 6: 端到端證據、瀏覽器驗收與操作文件

**Files:** Create `tests/integration/test_background_training_flow.py`, `tests/helpers/background_training_server.py`; modify `README.md`, `docs/training.md`, `history.md`。如需重用 fixtures，只在 `tests/conftest.py` 加入此流程的 tmp_path helper。

**Interfaces:** 消費 Tasks 1–5 已定義介面；不新增 production 介面。

- [ ] **6.1 寫真實小型 CPU 整合測試。** 使用既有 4 張 v1 train/validation fixture，將 validation 複製到測試 root/out/val-fixture，內建設定從 source resolver 讀取。以 epochs=1、batch_size=2、device=cpu 的內部 request 啟動真實 worker，等待最多 120 秒；驗證 task completed、checkpoint/report/bundle 存在、`validate_crop_bundle` 通過，result 指標與 report 相符，active-v1 sentinel 未改變。等待逾時先要求合作式取消、保留日誌，只清理本測試可識別的程序／tmp_path。

```python
from plateai_shared.bundle import validate_crop_bundle

def assert_training_outputs(task, root, source_root):
    assert task["status"] == "completed"
    assert task["progress"] == 100
    result = task["result"]
    assert Path(result["checkpoint"]).is_file()
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert result["final_accuracy"] == round(report["history"][-1]["val_acc"] * 100, 1)
    bundle = Path(result["bundle_dir"])
    assert bundle.parent == root / "models" / "bundles"
    validate_crop_bundle(bundle, source_root / "schemas" / "model_manifest.schema.json")
```

測試檔匯入 json/Path；使用 deadline 等待 task 終態後呼叫此 assertion helper。Task 4 的跨程序測試負責可重複的存活與競爭；此處專注真實數值和產物串接，不用大量資料拉長測試。
- [ ] **6.2 先跑此測試、修正整合差異，再跑一次完整測試。** `./.venv/Scripts/python.exe -m pytest tests/integration/test_background_training_flow.py -q`；接著 `./.venv/Scripts/python.exe -m pytest -q`。完整測試只在 Task 5 已隔離 Web 副作用後執行；預期全部通過，實際數量現場記錄，不沿用舊 history 的數字。失敗只修相關範圍，不能把既有 unrelated failure 宣稱為本次通過。
- [ ] **6.3 驗證 Windows server 生命週期。** 先確認目前 1688 及既有訓練狀態；不要終止別人的服務。測試用 server helper 接受 `--root`、`--port`、`--dev-reload`，將隔離資料根目錄放在只屬於 child 的 `PLATEAI_TEST_ROOT`，並設定 child 的 `CUDA_VISIBLE_DEVICES` 為空字串以固定 CPU。以 `uvicorn.run('tests.helpers.background_training_server:create_app', factory=True, host='127.0.0.1', port=args.port, reload=args.dev_reload)` 啟動，factory 如下；這些參數只存在測試 helper，不擴張產品 CLI。

```python
def create_app():
    import os
    import importlib
    from pathlib import Path
    web_module = importlib.import_module("plateai_web.app")
    web_module.ROOT = Path(os.environ["PLATEAI_TEST_ROOT"])
    return web_module.app
```

使用空閒測試 port，隔離 root/out 下放短資料 fixture，透過 HTTP 啟動足夠多 epochs 的 CPU 訓練，記 task ID、實際 worker PID/start_token、batch。分別測正常 Ctrl+C、關閉 Web 啟動主控台、`--dev-reload` 觸發 reload，再重新查詢原 ID，確認 batch 前進且可取消。真實主控台關閉若無法自動操作，留下待人工驗收項，不以 parent-exit helper 代替這項證據。實測必須記錄啟動環境是否有 Job Object 限制。
- [ ] **6.4 瀏覽器驗收。** 讀取當時可用的瀏覽器測試技能後，使用測試實例驗證：首 epoch 無 history 仍更新 batch；空 val 顯示尚未驗證；關頁重開接回；server 短暫離線顯示重連；stop 維持停止中直到 cancelled；未知 ID 不無限 404；開其他功能不打斷訓練輪詢；完成顯示匯出路徑且不宣稱已啟用。保存結果與 console/network 錯誤；未能執行的項目明列 pending。
- [ ] **6.5 更新操作文件。** README 說明 `run_server.bat` 與 `run_server.bat --dev-reload`、Web 關閉不停止訓練、從頁面停止；docs/training.md 列出 task DB/log 位置、階段/取消語義、電腦重開不續訓、run 與新 bundle 路徑、人工確認後才啟用模型。不得寫入自動覆寫 active-v1 的指令。history 記錄實際測試數、CPU/GPU 與 browser/console 的已驗證／未驗證範圍。
- [ ] **6.6 最後檢查交付。** 執行 `git -c core.whitespace=cr-at-eol diff --check`；檢查修改檔的 BOM/換行、`git status --short`、`git check-ignore runs/.web-training/tasks.sqlite3 models/bundles/train-example/manifest.json`。只列程式/測試/文件修改，不 stage 資料或產物；如使用者另要求 commit，逐一列出檔名 stage。

## Spec Coverage and Handoff

| Spec | 負責 Task | 必要證據 |
| --- | --- | --- |
| §1、§3 程序獨立且非排程平台 | 3、4、6 | parent-exit 測試 + 真實 Windows server 重啟 |
| §4 autocommit、單表、短 SQL | 1、4 | SQL trace、跨連線可見性、DB 故障停止 |
| §5 單筆互斥、pending/crash/身分核對 | 1、4、5 | 真實跨程序鎖、race/stale 狀態測試 |
| §6 真進度、取消與終態 | 2、3、4、5 | 首 batch/validation/cancel 測試 + UI |
| §7 輸出發布、保留模型 | 3、6 | 真實 export/parity、舊模型 sentinel |
| §8 API/launcher/UI 相容 | 5、6 | 隔離 API 回歸、CLI 旗標、瀏覽器 |
| §9 小範圍與依賴限制 | 全部 | diff、依賴未新增、其他任務回歸 |
| §10 各項驗收 | 1–6 | 逐項記錄，不將規格要求列成已完成 |

本 plan 的自檢包含 spec 覆蓋、介面一致性與 placeholder 檢查。推薦由目前任務 Native 依序實作，完成後集中檢查跨檔案資料流；若執行方式另有指定，以使用者選擇為準。這輪只交付 plan，不啟動上述測試、訓練、服務或程式修改。
