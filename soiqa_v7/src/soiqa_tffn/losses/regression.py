from __future__ import annotations

import torch.nn as nn


def build_loss(name: str = "mse") -> nn.Module:
    if name.lower() in {"mse", "l2", "euclidean"}:
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}")
