from __future__ import annotations

import torch
import torch.nn as nn

from soiqa_tffn.models.backbones.vmamba_like import VMambaLikeEncoder
from soiqa_tffn.models.norms import SafeLayerNorm


class PDIEBlock(nn.Module):
    def __init__(
        self,
        mf_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
        num_viewports: int = 20,
        concat_mode: str = "channel",
    ) -> None:
        super().__init__()
        self.num_viewports = int(num_viewports)
        self.concat_mode = str(concat_mode)
        if self.concat_mode not in {"channel", "grid"}:
            raise ValueError(f"Unsupported PDIE concat mode: {self.concat_mode}")

        in_channels = 3 * self.num_viewports if self.concat_mode == "channel" else 3
        self.viewport_encoder = VMambaLikeEncoder(
            in_channels=in_channels,
            embed_dim=128,
            out_dim=mf_dim,
            depth=depth,
            dropout=dropout,
        )
        self.norm = SafeLayerNorm(mf_dim)

    def _concat_diff_maps(self, diff: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = diff.shape
        if n != self.num_viewports:
            raise ValueError(f"PDIEBlock expected {self.num_viewports} viewports, but got {n}.")
        if self.concat_mode == "channel":
            return diff.reshape(b, n * c, h, w)

        grid_h = int(round(n ** 0.5))
        while grid_h > 1 and (n % grid_h) != 0:
            grid_h -= 1
        grid_w = n // grid_h
        diff = diff.view(b, grid_h, grid_w, c, h, w)
        diff = diff.permute(0, 3, 1, 4, 2, 5).contiguous()
        return diff.view(b, c, grid_h * h, grid_w * w)

    def forward(self, distorted_right: torch.Tensor, restored_right: torch.Tensor) -> torch.Tensor:
        diff = distorted_right - restored_right
        concat_diff = self._concat_diff_maps(diff)
        features = self.viewport_encoder(concat_diff)
        return self.norm(features)
