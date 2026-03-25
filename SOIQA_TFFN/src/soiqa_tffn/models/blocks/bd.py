from __future__ import annotations

import torch
import torch.nn as nn

from soiqa_tffn.models.blocks.attention import ShiftedWindowAttention2D
from soiqa_tffn.models.norms import SafeLayerNorm


class BinocularDifferenceBlock(nn.Module):
    """Hierarchical asymmetric BD block.

    This follows the paper's role more closely:
    - SW-MSA is only applied to the left branch by default.
    - the right branch uses LN/GELU style lightweight enhancement.
    - the current-stage difference is concatenated with previous-stage BD.
    - LN + FC are used to obtain the next hierarchical BD representation.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        prev_dim: int,
        num_heads: int = 8,
        window_size: int = 4,
        dropout: float = 0.1,
        double_side_swmsa: bool = False,
    ) -> None:
        super().__init__()
        self.left_attn = ShiftedWindowAttention2D(dim=dim, num_heads=num_heads, window_size=window_size, dropout=dropout)
        self.double_side_swmsa = bool(double_side_swmsa)
        self.right_attn = ShiftedWindowAttention2D(dim=dim, num_heads=num_heads, window_size=window_size, dropout=dropout)
        self.right_norm = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.in_dim = prev_dim + dim
        self.pre_norm = SafeLayerNorm(self.in_dim)
        self.fc = nn.Linear(self.in_dim, out_dim)
        self.out_norm = SafeLayerNorm(out_dim)

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor, prev_bd: torch.Tensor | None = None) -> torch.Tensor:
        enhanced_left = left_feat + self.act(self.left_attn(left_feat))
        if self.double_side_swmsa:
            enhanced_right = right_feat + self.act(self.right_attn(right_feat))
        else:
            enhanced_right = right_feat + self.act(self.right_norm(right_feat))

        diff_map = enhanced_left - enhanced_right
        diff_vec = diff_map.mean(dim=(2, 3))
        if prev_bd is None:
            concat = diff_vec
        else:
            concat = torch.cat([prev_bd, diff_vec], dim=-1)
        return self.out_norm(self.fc(self.pre_norm(concat)))
