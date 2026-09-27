"""Losses for crowd counting models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def match_tensor_shape(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Resize predictions to the target density-map resolution while preserving count."""
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction

    resized = F.interpolate(
        prediction,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    pred_sum = prediction.sum(dim=(-2, -1), keepdim=True)
    resized_sum = resized.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return resized * (pred_sum / resized_sum)


@dataclass(slots=True)
class LossBreakdown:
    total_loss: float
    density_loss: float
    count_loss: float


class CrowdCountingLoss(nn.Module):
    """Density-map loss with optional count-consistency regularization."""

    def __init__(
        self,
        density_weight: float = 1.0,
        count_weight: float = 0.0,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.density_weight = density_weight
        self.count_weight = count_weight
        self.density_criterion = nn.MSELoss(reduction=reduction)
        self.count_criterion = nn.L1Loss(reduction="mean")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, LossBreakdown]:
        prediction = match_tensor_shape(prediction, target)
        density_loss = self.density_criterion(prediction, target)
        pred_count = prediction.flatten(1).sum(dim=1)
        target_count = target.flatten(1).sum(dim=1)
        count_loss = self.count_criterion(pred_count, target_count)
        total_loss = (
            self.density_weight * density_loss
            + self.count_weight * count_loss
        )
        breakdown = LossBreakdown(
            total_loss=float(total_loss.detach().item()),
            density_loss=float(density_loss.detach().item()),
            count_loss=float(count_loss.detach().item()),
        )
        return total_loss, breakdown

