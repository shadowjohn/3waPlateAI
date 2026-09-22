"""Repository-owned single-class CSP-tiny/FPN/PAN pose detector.

Rows are P3, P4, then P5 in row-major grid order, with no NMS. Each decoded
row is cx, cy, width, height, confidence, LT.x, LT.y, RT.x, RT.y, RB.x,
RB.y, LB.x, LB.y in 640x640 letterbox pixels.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class _CSP(nn.Module):
    """Split feature flow into a short path and a residual spatial path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.short = _conv(in_channels, hidden, 1)
        self.main = _conv(in_channels, hidden, 1)
        self.residual = nn.Sequential(_conv(hidden, hidden, 1), _conv(hidden, hidden))
        self.fuse = _conv(2 * hidden, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        main = self.main(value)
        return self.fuse(torch.cat((self.short(value), main + self.residual(main)), dim=1))


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(_conv(3, 16, stride=2), _conv(16, 32, stride=2), _CSP(32, 32))
        self.p3 = nn.Sequential(_conv(32, 64, stride=2), _CSP(64, 64))
        self.p4 = nn.Sequential(_conv(64, 96, stride=2), _CSP(96, 96))
        self.p5 = nn.Sequential(_conv(96, 128, stride=2), _CSP(128, 128))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.p3(self.stem(images))
        p4 = self.p4(p3)
        return p3, p4, self.p5(p4)


class _Neck(nn.Module):
    """Top-down FPN followed by bottom-up PAN, all at 64 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.lateral4 = _conv(96, 64, 1)
        self.lateral5 = _conv(128, 64, 1)
        self.fpn4 = _CSP(128, 64)
        self.fpn3 = _CSP(128, 64)
        self.down3 = _conv(64, 64, stride=2)
        self.down4 = _conv(64, 64, stride=2)
        self.pan4 = _CSP(128, 64)
        self.pan5 = _CSP(128, 64)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features
        top5 = self.lateral5(p5)
        top4 = self.fpn4(torch.cat((self.lateral4(p4), F.interpolate(top5, scale_factor=2, mode="nearest")), dim=1))
        out3 = self.fpn3(torch.cat((p3, F.interpolate(top4, scale_factor=2, mode="nearest")), dim=1))
        out4 = self.pan4(torch.cat((self.down3(out3), top4), dim=1))
        out5 = self.pan5(torch.cat((self.down4(out4), top5), dim=1))
        return out3, out4, out5


def _validate_detector_images(images: torch.Tensor) -> None:
    if images.dtype != torch.float32 or images.ndim != 4 or tuple(images.shape[1:]) != (3, 640, 640):
        raise ValueError("PlatePoseNet expects float32 RGB [batch, 3, 640, 640]")


class PlatePoseNet(nn.Module):
    """Fixed spatial input, dynamic batch, decoded [batch, 8400, 13] output."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.neck = _Neck()
        self.heads = nn.ModuleList(nn.Conv2d(64, 13, 1) for _ in range(3))
        # Dense grids are overwhelmingly background. Start at a 1% prior;
        # this affects new training only, not loading existing checkpoints.
        for head in self.heads:
            nn.init.constant_(head.bias[4:5], -math.log(99.0))
        for stride in (8, 16, 32):
            axis = torch.arange(640 // stride, dtype=torch.float32)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            self.register_buffer(f"grid_{stride}", torch.stack((xx, yy), dim=-1).reshape(1, -1, 2), persistent=False)

    def _decode(self, raw: torch.Tensor, stride: int) -> torch.Tensor:
        values = raw.flatten(2).transpose(1, 2)
        grid = getattr(self, f"grid_{stride}")
        # Unbounded offsets let neighbouring positives predict the same centre
        # and preserve semantic corners without geometry-based reordering.
        centres = (values[..., :2] + grid + 0.5) * stride
        sizes = (F.softplus(values[..., 2:4]) + 1e-4) * stride
        confidence = values[..., 4:5].sigmoid()
        corners = (values[..., 5:].reshape(values.shape[0], -1, 4, 2) + grid.unsqueeze(2) + 0.5) * stride
        return torch.cat((centres, sizes, confidence, corners.flatten(2)), dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_detector_images(images)
        p3, p4, p5 = self.neck(self.backbone(images))
        return torch.cat((
            self._decode(self.heads[0](p3), 8),
            self._decode(self.heads[1](p4), 16),
            self._decode(self.heads[2](p5), 32),
        ), dim=1)
