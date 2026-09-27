"""Training and evaluation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from crowd_detection.config import ExperimentConfig
from crowd_detection.losses import CrowdCountingLoss
from crowd_detection.metrics import MetricTracker
from crowd_detection.utils.checkpointing import save_checkpoint
from crowd_detection.utils.io import ensure_dir, write_csv, write_json


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer.name == "sgd":
        return SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
    if config.optimizer.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: ExperimentConfig) -> MultiStepLR:
    if config.scheduler.name != "multistep":
        raise ValueError(f"Unsupported scheduler: {config.scheduler.name}")
    return MultiStepLR(
        optimizer,
        milestones=list(config.scheduler.milestones),
        gamma=config.scheduler.gamma,
    )


def _move_batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images = batch["image"].to(device, non_blocking=True)
    densities = batch["density"].to(device, non_blocking=True)
    image_paths = list(batch["image_path"])
    return images, densities, image_paths


def run_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn: CrowdCountingLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    description: str = "Eval",
) -> tuple[dict[str, float], MetricTracker]:
    training = optimizer is not None
    model.train(training)
    tracker = MetricTracker()
    autocast_enabled = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    progress = tqdm(loader, desc=description, unit="batch")
    for batch in progress:
        images, target, image_paths = _move_batch_to_device(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                prediction = model(images)
                loss, breakdown = loss_fn(prediction, target)

            if training:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        tracker.update(
            prediction=prediction.detach(),
            target=target.detach(),
            total_loss=breakdown.total_loss,
            density_loss=breakdown.density_loss,
            count_loss=breakdown.count_loss,
            image_paths=image_paths,
        )
        metrics = tracker.compute()
        progress.set_postfix(
            {
                "mae": f"{metrics['mae']:.2f}",
                "rmse": f"{metrics['rmse']:.2f}",
                "psnr": f"{metrics['psnr']:.2f}",
            }
        )

    return tracker.compute(), tracker


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    output_root = ensure_dir(Path(config.training.output_dir) / config.experiment_name)
    checkpoints_dir = ensure_dir(output_root / "checkpoints")

    loss_fn = CrowdCountingLoss(
        density_weight=config.loss.density_weight,
        count_weight=config.loss.count_weight,
        reduction=config.loss.reduction,
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    best_mae = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            amp=config.training.amp,
            grad_clip_norm=config.training.grad_clip_norm,
            description=f"Train {epoch:03d}",
        )
        val_metrics, tracker = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            amp=False,
            grad_clip_norm=None,
            description=f"Eval  {epoch:03d}",
        )
        scheduler.step()

        train_entry = {"epoch": epoch, **train_metrics}
        val_entry = {"epoch": epoch, **val_metrics}
        history["train"].append(train_entry)
        history["val"].append(val_entry)

        latest_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_mae": best_mae,
            "config": asdict(config),
        }
        save_checkpoint(latest_payload, checkpoints_dir / "last.pt")

        if epoch % config.training.save_every == 0:
            save_checkpoint(latest_payload, checkpoints_dir / f"epoch_{epoch:03d}.pt")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_payload = {
                **latest_payload,
                "best_mae": best_mae,
                "best_metrics": val_metrics,
            }
            save_checkpoint(best_payload, checkpoints_dir / "best.pt")
            write_csv(
                output_root / "best_sample_analysis.csv",
                [record.__dict__ for record in tracker.sample_records],
            )
            write_json(
                output_root / "best_density_breakdown.json",
                tracker.density_breakdown(),
            )
            write_json(output_root / "best_metrics.json", val_metrics)

        write_json(output_root / "history.json", history)

    return history


def evaluate_checkpoint(
    model: torch.nn.Module,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, float]:
    output_root = ensure_dir(output_dir)
    loss_fn = CrowdCountingLoss(
        density_weight=config.loss.density_weight,
        count_weight=config.loss.count_weight,
        reduction=config.loss.reduction,
    )
    metrics, tracker = run_epoch(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=None,
        description="Evaluation",
    )
    write_json(output_root / "evaluation_metrics.json", metrics)
    write_json(output_root / "evaluation_density_breakdown.json", tracker.density_breakdown())
    write_csv(output_root / "evaluation_sample_analysis.csv", [record.__dict__ for record in tracker.sample_records])
    return metrics
