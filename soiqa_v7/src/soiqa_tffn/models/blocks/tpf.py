from __future__ import annotations

import torch
import torch.nn as nn

from soiqa_tffn.models.backbones.resnet_stages import ResNet50Stages
from soiqa_tffn.models.blocks.attention import TokenSelfAttention
from soiqa_tffn.models.blocks.bd import BinocularDifferenceBlock
from soiqa_tffn.models.blocks.bs import BinocularSummationBlock
from soiqa_tffn.models.norms import SafeLayerNorm


class TPFBlock(nn.Module):
    def __init__(
        self,
        pretrained_resnet: bool = True,
        stage_channels: tuple[int, int, int] = (512, 1024, 2048),
        proj_dim: int = 256,
        stem_channels: int = 64,
        layer1_channels: int = 256,
        blocks_per_stage: int = 2,
        bd_dim: int = 256,
        bs_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 4,
        dropout: float = 0.1,
        num_viewports: int = 20,
        double_side_swmsa: bool = False,
    ) -> None:
        super().__init__()
        self.num_viewports = int(num_viewports)
        self.backbone = ResNet50Stages(
            pretrained=pretrained_resnet,
            stage_channels=stage_channels,
            proj_dim=proj_dim,
            stem_channels=stem_channels,
            layer1_channels=layer1_channels,
            blocks_per_stage=blocks_per_stage,
        )

        prev_bd_dims = [0, bd_dim, bd_dim]
        prev_bs_dims = [0, bs_dim, bs_dim * 2]
        self.bd_blocks = nn.ModuleList(
            [
                BinocularDifferenceBlock(
                    dim=proj_dim,
                    out_dim=bd_dim,
                    prev_dim=prev_bd_dims[i],
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                    double_side_swmsa=double_side_swmsa,
                )
                for i in range(3)
            ]
        )
        self.bs_blocks = nn.ModuleList(
            [BinocularSummationBlock(dim=proj_dim, out_dim=bs_dim, prev_dim=prev_bs_dims[i]) for i in range(3)]
        )

        self.bd_concat_dim = bd_dim * 3
        self.bs_concat_dim = bs_dim * 6
        self.bd_stage_mixer = TokenSelfAttention(dim=self.bd_concat_dim, num_heads=num_heads, dropout=dropout)
        self.bs_stage_mixer = TokenSelfAttention(dim=self.bs_concat_dim, num_heads=num_heads, dropout=dropout)
        self.bd_project = nn.Sequential(
            SafeLayerNorm(num_viewports * self.bd_concat_dim),
            nn.Linear(num_viewports * self.bd_concat_dim, bd_dim),
            nn.GELU(),
        )
        self.bs_project = nn.Sequential(
            SafeLayerNorm(num_viewports * self.bs_concat_dim),
            nn.Linear(num_viewports * self.bs_concat_dim, bs_dim * 3),
            nn.GELU(),
        )
        self.bd_out = SafeLayerNorm(bd_dim)
        self.bs_out = SafeLayerNorm(bs_dim * 3)

    def forward(self, left_viewports: torch.Tensor, right_viewports: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c, h, w = left_viewports.shape
        if n != self.num_viewports:
            raise ValueError(f"TPFBlock expected {self.num_viewports} viewports, but got {n}.")

        left_flat = left_viewports.view(b * n, c, h, w)
        right_flat = right_viewports.view(b * n, c, h, w)

        left_feats = self.backbone(left_flat)
        right_feats = self.backbone(right_flat)

        bd_prev = None
        bs_prev = None
        bd_stage_tokens: list[torch.Tensor] = []
        bs_stage_tokens: list[torch.Tensor] = []
        for stage_idx in range(3):
            bd_prev = self.bd_blocks[stage_idx](left_feats[stage_idx], right_feats[stage_idx], prev_bd=bd_prev)
            bs_prev = self.bs_blocks[stage_idx](left_feats[stage_idx], right_feats[stage_idx], prev_bs=bs_prev)
            bd_stage_tokens.append(bd_prev.view(b, n, -1))
            bs_stage_tokens.append(bs_prev.view(b, n, -1))

        bd_tokens = torch.cat(bd_stage_tokens, dim=-1)
        bs_tokens = torch.cat(bs_stage_tokens, dim=-1)
        bd_tokens = self.bd_stage_mixer(bd_tokens)
        bs_tokens = self.bs_stage_mixer(bs_tokens)

        bd = self.bd_project(bd_tokens.reshape(b, -1))
        bs = self.bs_project(bs_tokens.reshape(b, -1))
        return self.bd_out(bd), self.bs_out(bs)
