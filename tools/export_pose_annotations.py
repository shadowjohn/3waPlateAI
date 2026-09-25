"""Export reviewed coordinates only; never reads or copies image payloads."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from plateai_web.pose_data import export_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    doc = export_annotations(args.review, args.split, args.output)
    print(f"Exported {len(doc['records'])} reviewed records; no images, original labels, notes or paths.")


if __name__ == '__main__':
    main()
