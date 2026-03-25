from __future__ import annotations

import torch
import torch.nn as nn

from soiqa_tffn.models.norms import SafeLayerNorm


class DepthwiseSSMBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = SafeLayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pw1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pw2 = nn.Linear(dim * 2, dim)
        self.norm2 = SafeLayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.pw2(self.dropout(self.act(self.pw1(x))))
        x = residual + x
        x = x + self.ffn(self.norm2(x))
        return x


class VMambaLikeEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, out_dim: int = 256, depth: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([DepthwiseSSMBlock(embed_dim, dropout=dropout) for _ in range(depth)])
        self.norm = SafeLayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.fc(x)
