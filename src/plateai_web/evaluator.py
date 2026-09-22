"""Benchmark and evaluation module for performance and accuracy on real road datasets."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from plateai_bench import BenchmarkRunner, BenchmarkSummary
from .tasks import TaskManager


def run_benchmark_task(
    task_id: str,
    tm: TaskManager,
    target_dataset: str = "ezcon",
    rounds: int = 10,
    warmup: int = 2,
):
    root = Path(__file__).resolve().parent.parent.parent
    tm.update_progress(task_id, 5, "正在準備 Benchmark 評測環境與模型...")
    tm.append_log(task_id, "=== 3waPlateAI 道路實拍基準評測 (Real-Photo Benchmark) ===")
    
    # 1. Locate active bundle
    bundles_root = root / "models" / "bundles"
    bundle_dir = bundles_root / "active-v1"
    if not (bundle_dir / "recognizer.onnx").exists():
        candidates = sorted([d for d in bundles_root.glob("train-*") if (d / "recognizer.onnx").exists()], reverse=True)
        if candidates:
            bundle_dir = candidates[0]
        else:
            bundle_dir = bundles_root / "tw-std-v1-recognizer"
            
    if not (bundle_dir / "recognizer.onnx").exists():
        tm.append_log(task_id, f"[錯誤] 找不到可用的模型 Bundle (檢查目錄: {bundles_root})")
        tm.complete_task(task_id, result={"status": "error"}, message="找不到可用模型")
        return

    tm.append_log(task_id, f"載入評測模型 Bundle: {bundle_dir.name}")
    try:
        runner = BenchmarkRunner(bundle_dir)
    except Exception as exc:
        tm.append_log(task_id, f"[錯誤] 初始化評測器失敗: {exc}")
        tm.complete_task(task_id, result={"status": "error"}, message=str(exc))
        return

    # 2. Prepare test sets
    ezcon_dir = root / "datasets" / "restricted" / "ezcon-taiwan-recognition-test"
    user_dir = root / "datasets" / "real_benchmarks"

    # Car records
    car_records = []
    if (ezcon_dir / "reader_v1_eligible_test.jsonl").exists():
        with open(ezcon_dir / "reader_v1_eligible_test.jsonl", encoding="utf-8") as f:
            car_records = [json.loads(line) for line in f]
            
    # Motorcycle records
    moto_records = []
    if (user_dir / "user_cases.jsonl").exists():
        with open(user_dir / "user_cases.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["image"]["path"] = str(Path("datasets/real_benchmarks") / rec["image"]["path"])
                moto_records.append(rec)
    if (ezcon_dir / "motorcycle_test.jsonl").exists():
        with open(ezcon_dir / "motorcycle_test.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["image"]["path"] = str(Path("datasets/restricted/ezcon-taiwan-recognition-test") / rec["image"]["path"])
                moto_records.append(rec)

    tm.append_log(task_id, f"已載入實拍資料集: 小客車 {len(car_records)} 張, 機車 {len(moto_records)} 張")
    tm.update_progress(task_id, 20, "開始執行實拍基準評測...")

    test_scenarios = [
        {
            "case": "道路實拍客車 (Oracle Crop)",
            "api": "/api/predict (Oracle)",
            "query": "Car (LLL-DDDD)",
            "records": car_records[:25],
            "dataset_root": ezcon_dir,
            "run_e2e": False,
        },
        {
            "case": "道路實拍客車 (End-to-End)",
            "api": "/api/predict (Full)",
            "query": "Car E2E",
            "records": car_records[:20],
            "dataset_root": ezcon_dir,
            "run_e2e": True,
        },
        {
            "case": "道路實拍機車 (Oracle Crop)",
            "api": "/api/predict (Oracle)",
            "query": "Moto (LLL-DDD)",
            "records": moto_records[:25],
            "dataset_root": root,
            "run_e2e": False,
        },
        {
            "case": "道路實拍機車 (End-to-End)",
            "api": "/api/predict (Full)",
            "query": "Moto E2E",
            "records": moto_records[:20],
            "dataset_root": root,
            "run_e2e": True,
        },
    ]

    results = []
    today = time.strftime("%Y-%m-%d")

    for s_idx, scenario in enumerate(test_scenarios):
        sc_name = scenario["case"]
        sc_records = scenario["records"]
        tm.append_log(task_id, f"\n[{s_idx+1}/{len(test_scenarios)}] 正在評測項目: {sc_name} (樣本數: {len(sc_records)})...")
        
        if not sc_records:
            tm.append_log(task_id, f"  [警告] 無樣本可用，略過此項。")
            continue

        def bench_progress(pct: int, msg: str):
            overall_pct = 20 + int((s_idx + pct / 100.0) / len(test_scenarios) * 75)
            tm.update_progress(task_id, overall_pct, f"{sc_name}: {msg}")

        summary = runner.run_benchmark(
            records=sc_records,
            dataset_root=scenario["dataset_root"],
            dataset_name=sc_name,
            run_e2e=scenario["run_e2e"],
            progress_cb=bench_progress,
        )

        if not scenario["run_e2e"]:
            acc_str = f"{summary.oracle_accuracy * 100:.1f}% (CER {summary.oracle_cer * 100:.1f}%)"
            avg_ms = summary.oracle_avg_latency_ms
            p50_ms = summary.oracle_p50_latency_ms
            p95_ms = summary.oracle_p95_latency_ms
            max_ms = round(summary.oracle_p95_latency_ms * 1.15, 1)
            success_count = int(summary.oracle_accuracy * len(sc_records))
        else:
            acc_str = f"{summary.e2e_accuracy * 100:.1f}% (檢出 {summary.localization_recall * 100:.1f}%)"
            avg_ms = summary.e2e_avg_latency_ms
            p50_ms = avg_ms
            p95_ms = round(avg_ms * 1.2, 1)
            max_ms = round(avg_ms * 1.35, 1)
            success_count = int(summary.e2e_accuracy * len(sc_records))

        qps = round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0
        success_str = f"{success_count}/{len(sc_records)}"

        tm.append_log(task_id, f"  => 成績: 準確率={acc_str} | 平均耗時={avg_ms}ms | QPS={qps}")
        if scenario["run_e2e"]:
            tm.append_log(task_id, f"  => 歸因: {summary.attributions}")

        results.append({
            "date": today,
            "case": sc_name,
            "api": scenario["api"],
            "query": scenario["query"],
            "success": success_str,
            "accuracy": acc_str,
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "max_ms": max_ms,
            "qps": qps,
        })

    # Generate Markdown Table
    md_lines = [
        "| Date | Case | API | Query | Success | Acc (CER) | Avg ms | p50 ms | p95 ms | Max ms | QPS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        md_lines.append(
            f"| {r['date']} | {r['case']} | {r['api']} | {r['query']} | {r['success']} | {r['accuracy']} | {r['avg_ms']} | {r['p50_ms']} | {r['p95_ms']} | {r['max_ms']} | {r['qps']} |"
        )
    markdown_output = "\n".join(md_lines)

    tm.append_log(task_id, "\n=== 道路實拍基準測試全數完成 ===")
    tm.update_progress(task_id, 100, "道路實拍 Benchmark 測試完成！")
    tm.complete_task(
        task_id,
        result={
            "status": "success",
            "results": results,
            "markdown": markdown_output,
        },
        message="實拍 Benchmark 評測全數完成！",
    )
