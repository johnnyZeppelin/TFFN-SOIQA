from __future__ import annotations

import torch
import torch.nn as nn

from soiqa_tffn.models.blocks.attention import TokenSelfAttention
from soiqa_tffn.models.norms import SafeLayerNorm


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        bd_dim: int,
        bs_dim: int,
        mf_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        mode: str = "ff",
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"ff", "simple_concat"}:
            raise ValueError(f"Unsupported FF mode: {self.mode}")

        self.bd_enhance = TokenSelfAttention(dim=bd_dim, num_heads=num_heads, dropout=dropout)
        self.bs_norm = SafeLayerNorm(bs_dim)
        self.act = nn.GELU()
        self.out_dim = hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(bd_dim + bs_dim + mf_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, bd: torch.Tensor, bs: torch.Tensor, mf: torch.Tensor) -> torch.Tensor:
        if self.mode == "simple_concat":
            fused = torch.cat([bd, mf, bs], dim=-1)
            return self.fuse(fused)

        bdg = self.bd_enhance(bd.unsqueeze(1)).squeeze(1)
        bsg = bs + self.act(self.bs_norm(bs))
        fused = torch.cat([bdg, mf, bsg], dim=-1)
        return self.fuse(fused)
