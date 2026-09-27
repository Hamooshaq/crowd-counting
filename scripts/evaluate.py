#!/usr/bin/env python3
"""Evaluate a trained checkpoint and export Stage 2 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crowd_detection.config import load_experiment_config
from crowd_detection.data.dataset import build_dataloaders
from crowd_detection.engine.trainer import evaluate_checkpoint
from crowd_detection.models.csrnet import build_model
from crowd_detection.utils.checkpointing import load_checkpoint
from crowd_detection.utils.reproducibility import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-dir", default=None, help="Directory to store evaluation artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    set_seed(config.seed)
    device = resolve_device(config.training.device)
    print(f"Using device: {device}")
    _, val_loader = build_dataloaders(config.data)
    model = build_model(config.model).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model_state"])

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(config.training.output_dir) / config.experiment_name / "evaluation"
    )
    metrics = evaluate_checkpoint(
        model=model,
        loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    print(metrics)


if __name__ == "__main__":
    main()
