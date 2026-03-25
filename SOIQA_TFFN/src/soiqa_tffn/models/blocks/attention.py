from __future__ import annotations

import torch
import torch.nn as nn

from soiqa_tffn.models.norms import SafeLayerNorm


class ShiftedWindowAttention2D(nn.Module):
    """A dependency-light SW-MSA surrogate.

    We keep the same asymmetric role as the paper but replace exact windowed MHA
    with shifted depthwise token mixing for maximum runtime stability.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4, shift_size: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        _ = num_heads
        _ = window_size
        self.shift_size = 1 if shift_size is None else shift_size
        self.norm = nn.GroupNorm(1, dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = min(self.shift_size, max(x.shape[-2] - 1, 0), max(x.shape[-1] - 1, 0))
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))
        y = self.pwconv(self.dwconv(self.norm(x)))
        y = self.dropout(y)
        if shift > 0:
            y = torch.roll(y, shifts=(shift, shift), dims=(2, 3))
        return y


class TokenSelfAttention(nn.Module):
    """A stable token-mixing surrogate for self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        _ = num_heads
        self.norm = SafeLayerNorm(dim)
        self.token_mixer = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = self.channel_mlp(x)
        return residual + x
