#!/usr/bin/env python3
"""Fetch sample real Taiwan license plate photos for testing and evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen

DATASET_ID = "EZCon/taiwan-license-plate-recognition"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SPLIT = "test"
PAGE_SIZE = 100


def request_json(url: str, timeout: int = 20) -> dict:
    req = Request(url, headers={"User-Agent": "3waPlateAI-sample-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all_rows() -> list[dict]:
    print(f"Connecting to Hugging Face dataset ({DATASET_ID})...")
    first_page = request_json(
        f"{ROWS_URL}?dataset={DATASET_ID.replace('/', '%2F')}&config=default&split={SPLIT}&offset=0&length={PAGE_SIZE}"
    )
    total = first_page.get("num_rows_total", 0)
    rows = list(first_page.get("rows", []))
    print(f"Total available real test samples: {total}")

    for offset in range(PAGE_SIZE, total, PAGE_SIZE):
        page = request_json(
            f"{ROWS_URL}?dataset={DATASET_ID.replace('/', '%2F')}&config=default&split={SPLIT}&offset={offset}&length={PAGE_SIZE}"
        )
        rows.extend(page.get("rows", []))

    return rows


def download_single_image(row_item: dict, output_dir: Path) -> dict | None:
    idx = row_item.get("row_idx", 0)
    row = row_item.get("row", {})
    image_info = row.get("image", {})
    img_url = image_info.get("src")
    plate = row.get("license_number", "UNKNOWN").strip()
    is_ev = row.get("is_electric_car", False)
    xywhr = row.get("xywhr", [])

    if not img_url:
        return None

    # Safe filename: 002_MYX-6873.jpg
    safe_plate = re.sub(r'[\\/*?:"<>|]', "", plate)
    filename = f"{idx:03d}_{safe_plate}.jpg"
    target_path = output_dir / filename

    for attempt in range(3):
        try:
            req = Request(img_url, headers={"User-Agent": "3waPlateAI-sample-fetcher/1.0"})
            with urlopen(req, timeout=15) as res:
                content = res.read()
            target_path.write_bytes(content)
            break
        except Exception:
            if attempt == 2:
                print(f"[WARN] Failed to download {filename} after 3 attempts", file=sys.stderr)
                return None
            time.sleep(1)

    return {
        "filename": filename,
        "index": idx,
        "license_number": plate,
        "canonical": plate.replace("-", ""),
        "is_electric_car": is_ev,
        "xywhr": xywhr,
        "size_bytes": len(content),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download real Taiwan license plate sample photos for testing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_data"),
        help="Destination directory for sample photos (default: test_data).",
    )
    parser.add_argument(
        "--count",
        type=str,
        default="30",
        help="Number of sample images to fetch (e.g. 20, 50, or 'all'/'259'; default: 30).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8).",
    )
    args = parser.parse_args(argv)

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_all_rows()
    if not rows:
        print("[ERROR] Could not retrieve dataset rows.", file=sys.stderr)
        return 1

    count_str = str(args.count).strip().lower()
    if count_str in ("all", "max", "0"):
        target_count = len(rows)
    else:
        try:
            target_count = min(int(count_str), len(rows))
        except ValueError:
            target_count = 30

    selected_rows = rows[:target_count]
    print(f"Downloading {len(selected_rows)} real Taiwan license plate photos into: {out_dir}")

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_single_image, r, out_dir) for r in selected_rows]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                print(f"  [{len(results)}/{len(selected_rows)}] Downloaded {res['filename']} ({res['license_number']})")

    results.sort(key=lambda x: x["index"])

    # Write labels.json
    labels_file = out_dir / "labels.json"
    labels_data = {r["filename"]: r for r in results}
    labels_file.write_text(
        json.dumps(labels_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Write README.md
    readme_file = out_dir / "README.md"
    sample_example = f"test_data\\{results[0]['filename']}" if results else "test_data\\002_MYX-6873.jpg"
    readme_content = f"""# 3waPlateAI - 真實測試樣本圖片 (Sample Test Data)

本目錄包含從開源資料集 (EZCon/taiwan-license-plate-recognition) 取得之台灣實際車輛與車牌照片，供本機端辨識檢驗與測試使用。

## 樣本資訊
- **總張數**: {len(results)}
- **標註檔案**: `labels.json`
- **影像命名格式**: `{{序號}}_{{真實車牌號碼}}.jpg` (例如 `002_MYX-6873.jpg`)

## 如何測試辨識

### 1. 使用 CLI 命令辨識單張照片：
```powershell
.\\.venv\\Scripts\\plateai-read --bundle models\\bundles\\active-v1 {sample_example}
```

### 2. 使用 Web Studio 視覺化測試：
```powershell
.\\run_server.bat
```
在瀏覽器打開 http://localhost:1688 ，即可於「辨識測試」分頁直接拖拉或選取本目錄下的真實車牌照片進行高精度辨識測試！
"""
    readme_file.write_text(readme_content, encoding="utf-8")

    elapsed = time.time() - t0
    print("\n============================================================")
    print(f" Successfully downloaded {len(results)} sample photos in {elapsed:.2f}s!")
    print(f" Destination folder: {out_dir}")
    print(f" Ground truth labels: {labels_file}")
    print("============================================================")
    if results:
        sample_name = results[0]["filename"]
        print(f"Try running recognition on a sample image:")
        print(f"  .\\.venv\\Scripts\\plateai-read --bundle models\\bundles\\active-v1 {out_dir}\\{sample_name}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
