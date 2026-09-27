#!/usr/bin/env python3
"""Compare baseline and advanced experiment folders and emit a report-ready table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline output directory.")
    parser.add_argument("--advanced", required=True, help="Advanced-model output directory.")
    parser.add_argument("--output", default=None, help="Optional Markdown file path.")
    return parser.parse_args()


def load_metrics(root: Path) -> dict:
    candidates = [
        root / "best_metrics.json",
        root / "evaluation" / "evaluation_metrics.json",
        root / "evaluation_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"No metrics JSON found under {root}")


def render_table(baseline: dict, advanced: dict) -> str:
    rows = [
        ("MAE", baseline["mae"], advanced["mae"], advanced["mae"] - baseline["mae"]),
        ("RMSE", baseline["rmse"], advanced["rmse"], advanced["rmse"] - baseline["rmse"]),
        ("PSNR", baseline["psnr"], advanced["psnr"], advanced["psnr"] - baseline["psnr"]),
        ("SSIM", baseline["ssim"], advanced["ssim"], advanced["ssim"] - baseline["ssim"]),
    ]
    lines = [
        "| Metric | Baseline | Advanced | Delta (Advanced - Baseline) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, base_value, adv_value, delta in rows:
        lines.append(f"| {name} | {base_value:.4f} | {adv_value:.4f} | {delta:+.4f} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline)
    advanced_root = Path(args.advanced)
    baseline_metrics = load_metrics(baseline_root)
    advanced_metrics = load_metrics(advanced_root)
    table = render_table(baseline_metrics, advanced_metrics)
    print(table)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n")


if __name__ == "__main__":
    main()
