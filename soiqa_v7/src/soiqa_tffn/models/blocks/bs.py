from __future__ import annotations

import torch
import torch.nn as nn


class BinocularSummationBlock(nn.Module):
    """Hierarchical BS block following Eq. (4)."""

    def __init__(self, dim: int, out_dim: int, prev_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, out_dim)
        self.prev_dim = prev_dim

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor, prev_bs: torch.Tensor | None = None) -> torch.Tensor:
        left_vec = left_feat.mean(dim=(2, 3))
        right_vec = right_feat.mean(dim=(2, 3))
        summation = self.fc(left_vec + right_vec)
        if prev_bs is None:
            return summation
        return torch.cat([prev_bs, summation], dim=-1)
