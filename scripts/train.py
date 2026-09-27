#!/usr/bin/env python3
"""Train baseline or advanced crowd-counting models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crowd_detection.config import load_experiment_config
from crowd_detection.data.dataset import build_dataloaders
from crowd_detection.engine.trainer import train_model
from crowd_detection.models.csrnet import build_model
from crowd_detection.utils.io import ensure_dir, write_json
from crowd_detection.utils.reproducibility import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    set_seed(config.seed)
    device = resolve_device(config.training.device)
    print(f"Using device: {device}")
    train_loader, val_loader = build_dataloaders(config.data)
    model = build_model(config.model).to(device)

    output_root = ensure_dir(Path(config.training.output_dir) / config.experiment_name)
    write_json(output_root / "resolved_config.json", asdict(config))
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    write_json(output_root / "history.json", history)
    print(f"Training complete. Artifacts written to {output_root}")


if __name__ == "__main__":
    main()
