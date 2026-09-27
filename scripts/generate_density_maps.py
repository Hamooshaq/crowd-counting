#!/usr/bin/env python3
"""Generate ShanghaiTech density maps from point annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crowd_detection.data.density_maps import build_density_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing ShanghaiTech part_A/part_B folders.")
    parser.add_argument("--part", default="A", choices=["A", "B"], help="Dataset part to process.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to process, e.g. train test.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Maximum worker threads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    for split in args.splits:
        results = build_density_maps(root=root, part=args.part, split=split, max_workers=args.workers)
        mean_people = sum(int(entry["count"]) for entry in results) / len(results)
        mean_density = sum(float(entry["density_sum"]) for entry in results) / len(results)
        print(
            f"[part_{args.part}/{split}] images={len(results)} "
            f"mean_people={mean_people:.2f} mean_density_sum={mean_density:.2f}"
        )


if __name__ == "__main__":
    main()
