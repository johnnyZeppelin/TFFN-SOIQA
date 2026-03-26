from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


def _make_stage(in_ch: int, out_ch: int, stride: int, num_blocks: int) -> nn.Sequential:
    blocks = [ResidualBlock(in_ch, out_ch, stride=stride)]
    for _ in range(max(num_blocks - 1, 0)):
        blocks.append(ResidualBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


class ResNet50Stages(nn.Module):
    """A lightweight ResNet-style stage extractor.

    This keeps the same *role* as the paper's ResNet50 stage2/3/4 pipeline,
    while avoiding external torchvision dependency problems.
    """

    def __init__(
        self,
        pretrained: bool = False,
        stage_channels: tuple[int, int, int] = (512, 1024, 2048),
        proj_dim: int = 256,
        stem_channels: int = 64,
        layer1_channels: int = 256,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        _ = pretrained  # kept for API compatibility
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_stage(stem_channels, layer1_channels, stride=1, num_blocks=blocks_per_stage)
        self.layer2 = _make_stage(layer1_channels, stage_channels[0], stride=2, num_blocks=blocks_per_stage)
        self.layer3 = _make_stage(stage_channels[0], stage_channels[1], stride=2, num_blocks=blocks_per_stage)
        self.layer4 = _make_stage(stage_channels[1], stage_channels[2], stride=2, num_blocks=blocks_per_stage)

        self.proj2 = nn.Conv2d(stage_channels[0], proj_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(stage_channels[1], proj_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(stage_channels[2], proj_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        s2 = self.layer2(x)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return [self.proj2(s2), self.proj3(s3), self.proj4(s4)]
