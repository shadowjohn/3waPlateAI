"""The fixed v1 80-step temporal-convolution CTC recognizer."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A convolution, normalization, activation, and optional spatial pool."""

    def __init__(
        self, in_channels: int, out_channels: int, pool: tuple[int, int] | None
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(nn.MaxPool2d(pool))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TemporalDepthwiseBlock(nn.Module):
    """Residual depthwise temporal mixing with a dilation-specific receptive field."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.branch(value))


class PlateCTCNet(nn.Module):
    """A crop recognizer with a fixed 80-step CTC time axis."""

    def __init__(self, class_count: int = 34) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("PlateCTCNet requires at least 2 classes (blank + 1 symbol)")
        self.class_count = class_count
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 1)),
            ConvBlock(64, 128, pool=(2, 1)),
            nn.AdaptiveAvgPool2d((1, 80)),
        )
        self.temporal = nn.Sequential(
            TemporalDepthwiseBlock(128, dilation=1),
            TemporalDepthwiseBlock(128, dilation=2),
            TemporalDepthwiseBlock(128, dilation=4),
        )
        self.classifier = nn.Conv1d(128, class_count, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 64, 160):
            raise ValueError("PlateCTCNet expects input shaped [batch, 1, 64, 160]")
        encoded = self.encoder(images)
        temporal = self.temporal(encoded.squeeze(2))
        return self.classifier(temporal).transpose(1, 2)
