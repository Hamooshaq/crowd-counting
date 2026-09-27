"""Baseline and advanced CSRNet models."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from torchvision import models

from crowd_detection.config import ModelConfig
from crowd_detection.models.blocks import CBAMBlock, MultiScaleContextBlock, ResidualDilatedBlock


def make_layers(
    cfg: Iterable[int | str],
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    dilation_rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        padding = dilation_rate
        conv = nn.Conv2d(
            in_channels,
            int(value),
            kernel_size=3,
            padding=padding,
            dilation=dilation_rate,
        )
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(int(value)), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        in_channels = int(value)
    return nn.Sequential(*layers)


class CSRNetBackbone(nn.Module):
    def __init__(self, load_imagenet_weights: bool = True) -> None:
        super().__init__()
        self.frontend_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.frontend = make_layers(self.frontend_cfg)
        if load_imagenet_weights:
            self._load_vgg16_weights()

    def _load_vgg16_weights(self) -> None:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        frontend_state = self.frontend.state_dict()
        vgg_state = vgg.features.state_dict()
        for key in frontend_state.keys():
            if key in vgg_state and frontend_state[key].shape == vgg_state[key].shape:
                frontend_state[key] = vgg_state[key]
        self.frontend.load_state_dict(frontend_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frontend(x)


class CSRNetBaseline(nn.Module):
    def __init__(self, load_imagenet_weights: bool = True) -> None:
        super().__init__()
        self.backbone = CSRNetBackbone(load_imagenet_weights=load_imagenet_weights)
        self.backend = make_layers([512, 512, 512, 256, 128, 64], in_channels=512, dilation=True)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in list(self.backend.modules()) + [self.head]:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.backend(x)
        return self.head(x)


class AdvancedCSRNet(nn.Module):
    def __init__(
        self,
        load_imagenet_weights: bool = True,
        context_rates: tuple[int, ...] = (1, 2, 4, 8),
        attention_reduction: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = CSRNetBackbone(load_imagenet_weights=load_imagenet_weights)
        self.context = MultiScaleContextBlock(
            channels=512,
            rates=context_rates,
            attention_reduction=attention_reduction,
        )
        self.backend = nn.Sequential(
            ResidualDilatedBlock(512, 512, dilation=2, dropout=dropout),
            ResidualDilatedBlock(512, 512, dilation=4, dropout=dropout),
            ResidualDilatedBlock(512, 256, dilation=2, dropout=dropout),
            ResidualDilatedBlock(256, 128, dilation=2, dropout=dropout),
            ResidualDilatedBlock(128, 64, dilation=1, dropout=dropout),
            CBAMBlock(64, reduction=attention_reduction),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        modules_to_init = [self.context, self.backend, self.head]
        for root_module in modules_to_init:
            for module in root_module.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.context(x)
        x = self.backend(x)
        return self.head(x)


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "csrnet_baseline":
        return CSRNetBaseline(load_imagenet_weights=config.load_imagenet_weights)
    if config.name == "advanced_csrnet":
        return AdvancedCSRNet(
            load_imagenet_weights=config.load_imagenet_weights,
            context_rates=config.context_rates,
            attention_reduction=config.attention_reduction,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model name: {config.name}")
