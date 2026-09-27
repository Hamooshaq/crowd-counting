"""Metrics and sample analysis helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from crowd_detection.losses import match_tensor_shape


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = match_tensor_shape(prediction, target)
    target_max = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    pred_norm = torch.clamp(prediction / target_max, 0.0, 1.0)
    target_norm = torch.clamp(target / target_max, 0.0, 1.0)
    mse = F.mse_loss(pred_norm, target_norm, reduction="none").flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = match_tensor_shape(prediction, target)
    _, channels, height, width = prediction.shape
    kernel_size = max(3, min(11, height if height % 2 == 1 else height - 1, width if width % 2 == 1 else width - 1))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(kernel_size, 3)
    sigma = max(float(kernel_size) / 6.0, 1.0)

    window = _gaussian_kernel(kernel_size, sigma, prediction.device).expand(channels, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    pred = prediction
    tgt = target

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_tgt = F.conv2d(tgt, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_tgt_sq = mu_tgt.pow(2)
    mu_pred_tgt = mu_pred * mu_tgt

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_tgt_sq = F.conv2d(tgt * tgt, window, padding=padding, groups=channels) - mu_tgt_sq
    sigma_pred_tgt = F.conv2d(pred * tgt, window, padding=padding, groups=channels) - mu_pred_tgt

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_pred_tgt + c1) * (2 * sigma_pred_tgt + c2)
    denominator = (mu_pred_sq + mu_tgt_sq + c1) * (sigma_pred_sq + sigma_tgt_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-10)
    return ssim_map.flatten(1).mean(dim=1)


def density_band(count: float) -> str:
    if count < 50:
        return "low"
    if count < 150:
        return "medium"
    return "high"


@dataclass(slots=True)
class SampleRecord:
    image_path: str
    ground_truth_count: float
    predicted_count: float
    absolute_error: float
    relative_error: float
    density_band: str


@dataclass(slots=True)
class MetricTracker:
    total_loss: float = 0.0
    total_density_loss: float = 0.0
    total_count_loss: float = 0.0
    total_abs_error: float = 0.0
    total_sq_error: float = 0.0
    total_psnr: float = 0.0
    total_ssim: float = 0.0
    sample_count: int = 0
    sample_records: list[SampleRecord] = field(default_factory=list)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        total_loss: float,
        density_loss: float,
        count_loss: float,
        image_paths: list[str],
    ) -> None:
        prediction = match_tensor_shape(prediction, target)
        pred_counts = prediction.flatten(1).sum(dim=1)
        target_counts = target.flatten(1).sum(dim=1)
        abs_errors = (pred_counts - target_counts).abs()
        psnr_values = compute_psnr(prediction, target)
        ssim_values = compute_ssim(prediction, target)

        batch_size = prediction.shape[0]
        self.total_loss += total_loss
        self.total_density_loss += density_loss
        self.total_count_loss += count_loss
        self.total_abs_error += float(abs_errors.sum().item())
        self.total_sq_error += float((abs_errors**2).sum().item())
        self.total_psnr += float(psnr_values.sum().item())
        self.total_ssim += float(ssim_values.sum().item())
        self.sample_count += batch_size

        for index in range(batch_size):
            gt_count = float(target_counts[index].item())
            pred_count = float(pred_counts[index].item())
            abs_error = float(abs_errors[index].item())
            rel_error = 0.0 if gt_count == 0 else abs_error / gt_count
            self.sample_records.append(
                SampleRecord(
                    image_path=image_paths[index],
                    ground_truth_count=gt_count,
                    predicted_count=pred_count,
                    absolute_error=abs_error,
                    relative_error=rel_error,
                    density_band=density_band(gt_count),
                )
            )

    def compute(self) -> dict[str, float]:
        if self.sample_count == 0:
            return {
                "loss": 0.0,
                "density_loss": 0.0,
                "count_loss": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "psnr": 0.0,
                "ssim": 0.0,
            }

        return {
            "loss": self.total_loss / self.sample_count,
            "density_loss": self.total_density_loss / self.sample_count,
            "count_loss": self.total_count_loss / self.sample_count,
            "mae": self.total_abs_error / self.sample_count,
            "rmse": math.sqrt(self.total_sq_error / self.sample_count),
            "psnr": self.total_psnr / self.sample_count,
            "ssim": self.total_ssim / self.sample_count,
        }

    def density_breakdown(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[SampleRecord]] = {"low": [], "medium": [], "high": []}
        for record in self.sample_records:
            buckets[record.density_band].append(record)

        summary: dict[str, dict[str, float]] = {}
        for band, records in buckets.items():
            if not records:
                summary[band] = {"count": 0, "mae": 0.0, "mean_relative_error": 0.0}
                continue
            summary[band] = {
                "count": len(records),
                "mae": sum(r.absolute_error for r in records) / len(records),
                "mean_relative_error": sum(r.relative_error for r in records) / len(records),
            }
        return summary

