"""CLI tool to run empirical real-photo benchmark separating Oracle OCR from End-to-End."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plateai_bench import BenchmarkRunner, BenchmarkSummary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 3waPlateAI models on real road photo benchmarks.")
    parser.add_argument(
        "--bundle",
        type=str,
        default="models/bundles/active-v1",
        help="Path to the model bundle to evaluate",
    )
    parser.add_argument(
        "--split",
        choices=["cars", "motorcycles", "user", "all"],
        default="all",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=["oracle", "e2e", "both"],
        default="both",
        help="Evaluation mode (oracle crop, end-to-end, or both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/benchmarks/baseline_report.json",
        help="Path to save the benchmark JSON report",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference execution device",
    )
    return parser


def load_split_records(root: Path, split: str) -> tuple[list[dict], Path]:
    ezcon_dir = root / "datasets" / "restricted" / "ezcon-taiwan-recognition-test"
    user_dir = root / "datasets" / "real_benchmarks"

    records = []
    dataset_root = root

    if split == "cars":
        records_path = ezcon_dir / "reader_v1_eligible_test.jsonl"
        dataset_root = ezcon_dir
        with open(records_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    elif split == "motorcycles":
        records_path = ezcon_dir / "motorcycle_test.jsonl"
        dataset_root = ezcon_dir
        with open(records_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    elif split == "user":
        records_path = user_dir / "user_cases.jsonl"
        dataset_root = user_dir
        with open(records_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    elif split == "all":
        # Combine user cases, ezcon cars, ezcon motorcycles
        if (user_dir / "user_cases.jsonl").exists():
            with open(user_dir / "user_cases.jsonl", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    # Adjust image path relative to ROOT
                    rec["image"]["path"] = str(Path("datasets/real_benchmarks") / rec["image"]["path"])
                    records.append(rec)
        if (ezcon_dir / "all_test.jsonl").exists():
            with open(ezcon_dir / "all_test.jsonl", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    rec["image"]["path"] = str(Path("datasets/restricted/ezcon-taiwan-recognition-test") / rec["image"]["path"])
                    records.append(rec)
        dataset_root = root
    else:
        raise ValueError(f"Unknown split: {split}")

    return records, dataset_root


def print_summary_table(summary: BenchmarkSummary, mode: str) -> None:
    print("\n" + "=" * 68)
    print(f"   3waPlateAI 道路實拍基準測試報告 (Real-Photo Benchmark)")
    print(f"   評測資料集: {summary.dataset_name} ({summary.sample_count} 張實拍樣本)")
    print(f"   受測模型 Bundle: {summary.bundle_name}")
    print("=" * 68)

    print("\n[Mode A: Oracle Crop OCR 純識別性能 (排除定位偏差)]")
    print(f"  • 整牌全對率 (Exact Match Acc):  {summary.oracle_accuracy * 100:.2f}%")
    print(f"  • 字元錯誤率 (CER):              {summary.oracle_cer * 100:.2f}%")
    print(f"  • 字元準確率 (Char Acc):         {summary.oracle_char_accuracy * 100:.2f}%")
    print(f"  • 平均延遲 (Avg Latency):        {summary.oracle_avg_latency_ms:.1f} ms (P50: {summary.oracle_p50_latency_ms:.1f} ms, P95: {summary.oracle_p95_latency_ms:.1f} ms)")

    if mode in ("e2e", "both"):
        print("\n[Mode B: End-to-End 定位+識別全流程實戰性能]")
        print(f"  • 車牌檢出率 (IoU >= 0.5):       {summary.localization_recall * 100:.2f}%")
        print(f"  • 端到端全對率 (E2E Exact Acc):  {summary.e2e_accuracy * 100:.2f}%")
        print(f"  • 端到端字元錯誤率 (E2E CER):    {summary.e2e_cer * 100:.2f}%")
        print(f"  • 全流程平均耗時:                {summary.e2e_avg_latency_ms:.1f} ms")

        print("\n[失誤歸因矩陣 (Failure Attribution Breakdown)]")
        total = max(1, summary.sample_count)
        for cat, cnt in summary.attributions.items():
            pct = cnt / total * 100
            bar = "█" * int(pct / 5)
            print(f"  • {cat:20s}: {cnt:4d} 件 ({pct:5.1f}%) {bar}")

    if summary.failures:
        print("\n[典型失敗案例分析 (前 8 件)]")
        for idx, item in enumerate(summary.failures[:8]):
            print(
                f"  [{idx+1:02d}] 圖片: {Path(item['image_path']).name} | "
                f"GT: {item['gt_canonical']:10s} -> "
                f"Oracle: {item['oracle_canonical']:10s} | "
                f"E2E: {item['e2e_canonical']:10s} (IoU={item['e2e_iou']:.2f}) | "
                f"歸因: {item['attribution']}"
            )
    print("=" * 68 + "\n")


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = ROOT / args.bundle if not Path(args.bundle).is_absolute() else Path(args.bundle)
    if not bundle_path.exists():
        print(f"Error: Bundle directory not found at {bundle_path}")
        sys.exit(1)

    records, dataset_root = load_split_records(ROOT, args.split)
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    print(f"載入評測資料集 [{args.split}]: 共 {len(records)} 筆真實道路標註")
    runner = BenchmarkRunner(bundle_path, device=args.device)

    def progress_cb(pct: int, msg: str) -> None:
        print(f"[{pct:3d}%] {msg}", flush=True)

    summary = runner.run_benchmark(
        records=records,
        dataset_root=dataset_root,
        dataset_name=f"real_photo_{args.split}",
        run_e2e=(args.mode in ("e2e", "both")),
        progress_cb=progress_cb,
    )

    print_summary_table(summary, args.mode)

    out_file = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"完整報告已成功寫入: {out_file}")


if __name__ == "__main__":
    main()
