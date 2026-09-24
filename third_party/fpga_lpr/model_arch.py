"""Minimal model architecture from evan6007/FPGA-LPR, pinned upstream commit.

Derived from model_utils.py at 574667ca7f5730d17b4b6fcda3ec568521bcbcd8.
Notebook/data-loader imports and commented experiments were omitted. See
UPSTREAM.md and MIT-LICENSE.txt. Developer-side conversion only; shipped
inference imports the ONNX pair, not this PyTorch module.
"""

from __future__ import annotations

import torch
from torch import nn


CHARS = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-")


class CPMStage_x(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=9, padding=4)
        self.bn4 = nn.BatchNorm2d(out_channels * 2)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu3(self.bn3(self.pool3(self.conv3(x))))
        x = self.relu4(self.bn4(self.conv4(x)))
        return self.relu5(self.bn5(self.conv5(x)))


class CPMStage_g1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.relu3(self.bn3(self.conv3(x)))


class CPMStage_g2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels * 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.relu3(self.bn3(self.conv3(x)))


class CPMLicensePlateNet(nn.Module):
    def __init__(self, num_stages: int = 6) -> None:
        super().__init__()
        self.CPMStage_x_1 = CPMStage_x(3, 32)
        self.CPMStage_g1_1 = CPMStage_g1(32, 4)
        self.CPMStage_x_2 = CPMStage_x(3, 32)
        self.CPMStage_g2_2 = CPMStage_g2(36, 4)
        self.CPMStage_x_3 = CPMStage_x(3, 32)
        self.CPMStage_g2_3 = CPMStage_g2(36, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stage = self.CPMStage_g1_1(self.CPMStage_x_1(x))
        next_features = self.CPMStage_x_2(x)
        heatmap = self.CPMStage_g2_2(torch.cat([stage, next_features], dim=1))
        return stage, heatmap


class small_basic_block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len: int, phase: bool, class_num: int, dropout_rate: float) -> None:
        super().__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            small_basic_block(ch_in=128, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(448 + self.class_num, self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(37, 37, kernel_size=(4, 1), stride=1, padding=(0, 0), groups=37),
        )
        self.c0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(9, 9), stride=(1, 2)),
            nn.Conv2d(64, 64, kernel_size=(5, 3), stride=(1, 2)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(9, 9), stride=(1, 2)),
            nn.Conv2d(128, 128, kernel_size=(5, 3), stride=(1, 2)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=128),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(9, 5), stride=(1, 2)),
            nn.Conv2d(256, 256, kernel_size=(5, 3), stride=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=256),
        )
        self.l1 = nn.Sequential(nn.Conv2d(37, 37, kernel_size=(25, 1), stride=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep_features = []
        for index, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if index in (2, 6, 13, 22):
                keep_features.append(x)
        global_context = []
        for index, feature in enumerate(keep_features):
            if index == 0:
                feature = self.c0(feature)
            elif index == 1:
                feature = self.c1(feature)
            elif index == 2:
                feature = self.c2(feature)
            global_context.append(feature)
        x = torch.cat(global_context, dim=1)
        x = self.container(x)
        x = self.l1(x)
        return x.reshape(x.shape[0], 37, 18)
