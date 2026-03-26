from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from soiqa_tffn.models.blocks.ff import FeatureFusionBlock
from soiqa_tffn.models.norms import SafeLayerNorm
from soiqa_tffn.models.blocks.pdie import PDIEBlock
from soiqa_tffn.models.blocks.tpf import TPFBlock


class TFFNModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        dataset_cfg = cfg["dataset"]
        self.bd_dim = int(model_cfg["bd_dim"])
        self.bs_dim = int(model_cfg["bs_dim"]) * 3
        self.mf_dim = int(model_cfg["mf_dim"])

        self.use_tpf_bd = bool(model_cfg.get("use_tpf_bd", True))
        self.use_tpf_bs = bool(model_cfg.get("use_tpf_bs", True))
        self.use_pdie = bool(model_cfg.get("use_pdie", True))

        self.tpf = TPFBlock(
            pretrained_resnet=bool(model_cfg.get("pretrained_resnet", True)),
            stage_channels=tuple(int(x) for x in model_cfg.get("stage_channels", [512, 1024, 2048])),
            proj_dim=int(model_cfg["proj_dim"]),
            stem_channels=int(model_cfg.get("stem_channels", 64)),
            layer1_channels=int(model_cfg.get("layer1_channels", 256)),
            blocks_per_stage=int(model_cfg.get("blocks_per_stage", 2)),
            bd_dim=int(model_cfg["bd_dim"]),
            bs_dim=int(model_cfg["bs_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            window_size=int(model_cfg["window_size"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            num_viewports=int(dataset_cfg.get("num_viewports", 20)),
            double_side_swmsa=bool(model_cfg.get("bd_double_side_swmsa", False)),
        )
        self.pdie = PDIEBlock(
            mf_dim=int(model_cfg["mf_dim"]),
            depth=int(model_cfg.get("pdie_depth", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            num_viewports=int(dataset_cfg.get("num_viewports", 20)),
            concat_mode=str(model_cfg.get("pdie_concat_mode", "grid")),
            encoder_type=str(model_cfg.get("pdie_encoder", "official_vmamba")),
            vmamba_cfg=dict(model_cfg.get("vmamba", {})),
        )
        self.ff = FeatureFusionBlock(
            bd_dim=self.bd_dim,
            bs_dim=self.bs_dim,
            mf_dim=self.mf_dim,
            num_heads=int(model_cfg["num_heads"]),
            hidden_dim=int(model_cfg["ff_hidden_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            mode=str(model_cfg.get("ff_mode", "ff")),
        )
        self.head = nn.Sequential(
            SafeLayerNorm(int(model_cfg["ff_hidden_dim"])),
            nn.Linear(int(model_cfg["ff_hidden_dim"]), 1),
        )

    def _zero_branch(self, ref: torch.Tensor, dim: int) -> torch.Tensor:
        return ref.new_zeros((ref.shape[0], dim))

    def forward(self, left_viewports: torch.Tensor, right_viewports: torch.Tensor, right_restored_viewports: torch.Tensor) -> torch.Tensor:
        bd, bs = self.tpf(left_viewports, right_viewports)
        mf = self.pdie(right_viewports, right_restored_viewports)

        if not self.use_tpf_bd:
            bd = self._zero_branch(bd, self.bd_dim)
        if not self.use_tpf_bs:
            bs = self._zero_branch(bs, self.bs_dim)
        if not self.use_pdie:
            mf = self._zero_branch(mf, self.mf_dim)

        fused = self.ff(bd, bs, mf)
        score = self.head(fused).squeeze(-1)
        return score


def build_model(cfg: dict[str, Any]) -> TFFNModel:
    return TFFNModel(cfg)
