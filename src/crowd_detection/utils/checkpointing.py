"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(payload: dict, path: str | Path) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(Path(path), map_location=device)

