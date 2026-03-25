from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from soiqa_tffn.config import save_yaml
from soiqa_tffn.data.score_normalizer import ScoreNormalizer
from soiqa_tffn.engine.checkpoint import save_checkpoint
from soiqa_tffn.engine.evaluator import evaluate_model
from soiqa_tffn.engine.schedulers import build_scheduler


def _build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    name = str(train_cfg["optimizer"]).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(train_cfg.get("momentum", 0.9)), weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def _select_device(cfg: dict[str, Any]) -> torch.device:
    requested_device = str(cfg["project"].get("device", "auto"))
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


def _append_history_row(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _is_better(current: float, best: float, monitor_mode: str, min_delta: float = 0.0) -> bool:
    if monitor_mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def train_model(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    train_loader,
    test_loader,
    criterion,
    logger,
    score_normalizer: ScoreNormalizer | None = None,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = _select_device(cfg)
    output_dir = Path(cfg["project"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    pred_dir = output_dir / "preds"
    history_path = output_dir / "history.csv"
    config_snapshot_path = output_dir / "resolved_config.yaml"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, config_snapshot_path)

    if score_normalizer is None:
        score_normalizer = ScoreNormalizer(mode="none")
    score_normalizer.save(output_dir / "score_normalizer.json")

    model.to(device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    apply_logistic_fit = bool(cfg.get("eval", {}).get("apply_logistic_fit", False))

    early_cfg = cfg.get("early_stop", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    monitor_key = str(early_cfg.get("monitor", "SRCC"))
    monitor_mode = str(early_cfg.get("mode", "max")).lower()

    start_epoch = 1
    best_metrics: dict[str, Any] = {}
    best_monitor = float("inf") if monitor_mode == "min" else float("-inf")
    stale_epochs = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=True)
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_metrics = dict(resume_state.get("best_metrics", {}))
        best_monitor = float(best_metrics.get(monitor_key, best_monitor))
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        logger.info(f"Resumed training from epoch {start_epoch}.")

    logger.info(f"Score normalization mode: {score_normalizer.mode}")
    logger.info(f"Eval logistic fitting: {apply_logistic_fit}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for batch in pbar:
            left = batch["left_viewports"].to(device, non_blocking=True)
            right = batch["right_viewports"].to(device, non_blocking=True)
            right_restored = batch["right_restored_viewports"].to(device, non_blocking=True)
            target = batch["score"].to(device, non_blocking=True)
            train_target = score_normalizer.transform_tensor(target) if score_normalizer.is_enabled else target

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(left, right, right_restored)
                loss = criterion(pred, train_target)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        pred_path = pred_dir / f"epoch_{epoch:03d}_test_predictions.csv"
        analysis_path = pred_dir / f"epoch_{epoch:03d}_test_analysis.json"
        metrics = evaluate_model(
            model,
            test_loader,
            device=device,
            save_predictions_path=pred_path,
            save_analysis_path=analysis_path,
            score_normalizer=score_normalizer,
            apply_logistic_fit=apply_logistic_fit,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time_sec = float(time.time() - epoch_start)

        logger.info(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | lr={current_lr:.6e} | "
            f"PLCC={metrics['PLCC']:.4f} SRCC={metrics['SRCC']:.4f} KRCC={metrics['KRCC']:.4f} RMSE={metrics['RMSE']:.4f} | "
            f"time={epoch_time_sec:.2f}s"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if use_amp else None,
            "metrics": metrics,
            "best_metrics": best_metrics,
            "stale_epochs": stale_epochs,
            "config": cfg,
            "score_normalizer": score_normalizer.to_dict(),
        }
        save_checkpoint(state, ckpt_dir / f"epoch_{epoch:03d}.pt")

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "PLCC": metrics["PLCC"],
            "SRCC": metrics["SRCC"],
            "KRCC": metrics["KRCC"],
            "RMSE": metrics["RMSE"],
            "PLCC_raw": metrics["PLCC_raw"],
            "RMSE_raw": metrics["RMSE_raw"],
            "lr": current_lr,
            "epoch_time_sec": epoch_time_sec,
        }
        _append_history_row(history_path, history_row)

        current_monitor = float(metrics[monitor_key])
        should_update_best = _is_better(current_monitor, best_monitor, monitor_mode=monitor_mode, min_delta=min_delta)
        if not best_metrics:
            should_update_best = True

        if should_update_best:
            if not (current_monitor != current_monitor):
                best_monitor = current_monitor
            best_metrics = {k: v for k, v in metrics.items() if k != "analysis"}
            stale_epochs = 0
            state["best_metrics"] = best_metrics
            state["stale_epochs"] = stale_epochs
            save_checkpoint(state, ckpt_dir / "best.pt")
            (pred_dir / "best_metrics.json").write_text(
                json.dumps({**best_metrics, "monitor": monitor_key, "monitor_mode": monitor_mode}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            stale_epochs += 1

        if early_enabled and stale_epochs >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} with patience={patience}.")
            break

    logger.info(f"Best test metrics: {best_metrics}")
    return best_metrics
