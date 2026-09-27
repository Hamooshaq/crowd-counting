"""Configuration loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    root: str
    part: str
    train_split: str
    val_split: str
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    resize: tuple[int, int] | None
    crop_size: tuple[int, int] | None
    horizontal_flip: bool
    downsample_factor: int


@dataclass(slots=True)
class ModelConfig:
    name: str
    load_imagenet_weights: bool
    dropout: float
    context_rates: tuple[int, ...]
    attention_reduction: int


@dataclass(slots=True)
class LossConfig:
    density_weight: float
    count_weight: float
    reduction: str


@dataclass(slots=True)
class OptimizerConfig:
    name: str
    lr: float
    momentum: float
    weight_decay: float


@dataclass(slots=True)
class SchedulerConfig:
    name: str
    milestones: tuple[int, ...]
    gamma: float


@dataclass(slots=True)
class TrainingConfig:
    epochs: int
    device: str
    amp: bool
    grad_clip_norm: float | None
    output_dir: str
    save_every: int


@dataclass(slots=True)
class EvaluationConfig:
    save_predictions: bool
    sample_count: int


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _as_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"Expected a size pair, got {value!r}")
    return int(value[0]), int(value[1])


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text())

    return ExperimentConfig(
        seed=int(payload["seed"]),
        experiment_name=str(payload["experiment_name"]),
        data=DataConfig(
            root=str(payload["data"]["root"]),
            part=str(payload["data"]["part"]),
            train_split=str(payload["data"]["train_split"]),
            val_split=str(payload["data"]["val_split"]),
            train_batch_size=int(payload["data"]["train_batch_size"]),
            eval_batch_size=int(payload["data"]["eval_batch_size"]),
            num_workers=int(payload["data"]["num_workers"]),
            pin_memory=bool(payload["data"]["pin_memory"]),
            resize=_as_size(payload["data"]["resize"]),
            crop_size=_as_size(payload["data"]["crop_size"]),
            horizontal_flip=bool(payload["data"]["horizontal_flip"]),
            downsample_factor=int(payload["data"]["downsample_factor"]),
        ),
        model=ModelConfig(
            name=str(payload["model"]["name"]),
            load_imagenet_weights=bool(payload["model"]["load_imagenet_weights"]),
            dropout=float(payload["model"]["dropout"]),
            context_rates=tuple(int(rate) for rate in payload["model"]["context_rates"]),
            attention_reduction=int(payload["model"]["attention_reduction"]),
        ),
        loss=LossConfig(
            density_weight=float(payload["loss"]["density_weight"]),
            count_weight=float(payload["loss"]["count_weight"]),
            reduction=str(payload["loss"]["reduction"]),
        ),
        optimizer=OptimizerConfig(
            name=str(payload["optimizer"]["name"]),
            lr=float(payload["optimizer"]["lr"]),
            momentum=float(payload["optimizer"]["momentum"]),
            weight_decay=float(payload["optimizer"]["weight_decay"]),
        ),
        scheduler=SchedulerConfig(
            name=str(payload["scheduler"]["name"]),
            milestones=tuple(int(step) for step in payload["scheduler"]["milestones"]),
            gamma=float(payload["scheduler"]["gamma"]),
        ),
        training=TrainingConfig(
            epochs=int(payload["training"]["epochs"]),
            device=str(payload["training"]["device"]),
            amp=bool(payload["training"]["amp"]),
            grad_clip_norm=(
                None
                if payload["training"]["grad_clip_norm"] is None
                else float(payload["training"]["grad_clip_norm"])
            ),
            output_dir=str(payload["training"]["output_dir"]),
            save_every=int(payload["training"]["save_every"]),
        ),
        evaluation=EvaluationConfig(
            save_predictions=bool(payload["evaluation"]["save_predictions"]),
            sample_count=int(payload["evaluation"]["sample_count"]),
        ),
    )

