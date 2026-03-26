from __future__ import annotations

import math
from typing import Any

import torch


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg.get("scheduler", {})
    name = str(scheduler_cfg.get("name", "none")).lower()
    epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

    if name == "none":
        return None

    if name == "cosine":
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

        def lr_lambda(current_epoch: int) -> float:
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            progress_denom = max(1, epochs - warmup_epochs)
            progress = float(current_epoch - warmup_epochs) / float(progress_denom)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if warmup_epochs > 0:
            initial_factor = lr_lambda(0)
            for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
                group["lr"] = base_lr * initial_factor
        return scheduler

    if name == "multistep":
        milestones = [int(x) for x in scheduler_cfg.get("milestones", [30, 40])]
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    raise ValueError(f"Unsupported scheduler: {name}")
