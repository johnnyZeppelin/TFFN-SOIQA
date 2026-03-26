from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from soiqa_tffn.data.score_normalizer import ScoreNormalizer
from soiqa_tffn.metrics import compute_iqa_metrics


def _build_prediction_frame(
    all_names: list[str],
    all_targets: list[float],
    all_preds: list[float],
    all_contents: list[str],
    all_dist_types: list[str],
    all_dist_levels: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_name": all_names,
            "target": all_targets,
            "prediction": all_preds,
            "content_name": all_contents,
            "distortion_type": all_dist_types,
            "distortion_level": all_dist_levels,
        }
    )


def _group_metrics(df: pd.DataFrame, group_col: str, apply_logistic_fit: bool) -> dict[str, Any]:
    if group_col not in df.columns:
        return {}
    result: dict[str, Any] = {}
    for key, part in df.groupby(group_col, dropna=False):
        group_name = "NA" if pd.isna(key) else str(key)
        metrics = compute_iqa_metrics(part["target"].to_numpy(), part["prediction"].to_numpy(), apply_logistic_fit=apply_logistic_fit)
        result[group_name] = {"count": int(len(part)), **metrics}
    return result


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    save_predictions_path: str | Path | None = None,
    save_analysis_path: str | Path | None = None,
    score_normalizer: ScoreNormalizer | None = None,
    apply_logistic_fit: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_targets: list[float] = []
    all_preds: list[float] = []
    all_names: list[str] = []
    all_contents: list[str] = []
    all_dist_types: list[str] = []
    all_dist_levels: list[str] = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        left = batch["left_viewports"].to(device, non_blocking=True)
        right = batch["right_viewports"].to(device, non_blocking=True)
        right_restored = batch["right_restored_viewports"].to(device, non_blocking=True)
        target = batch["score"].to(device, non_blocking=True)

        pred = model(left, right, right_restored)
        if score_normalizer is not None and score_normalizer.is_enabled:
            pred = score_normalizer.inverse_tensor(pred)

        all_targets.extend(target.detach().cpu().tolist())
        all_preds.extend(pred.detach().cpu().tolist())
        all_names.extend(batch["image_name"])
        all_contents.extend([str(x) for x in batch.get("content_name", [""] * len(batch["image_name"]))])
        all_dist_types.extend([str(x) for x in batch.get("distortion_type", [""] * len(batch["image_name"]))])
        all_dist_levels.extend([str(x) for x in batch.get("distortion_level", [""] * len(batch["image_name"]))])

    metrics = compute_iqa_metrics(all_targets, all_preds, apply_logistic_fit=apply_logistic_fit)
    pred_df = _build_prediction_frame(
        all_names=all_names,
        all_targets=all_targets,
        all_preds=all_preds,
        all_contents=all_contents,
        all_dist_types=all_dist_types,
        all_dist_levels=all_dist_levels,
    )

    if save_predictions_path is not None:
        save_predictions_path = Path(save_predictions_path)
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(save_predictions_path, index=False)

    analysis = {
        "overall": {"count": int(len(pred_df)), **metrics},
        "by_distortion_type": _group_metrics(pred_df, group_col="distortion_type", apply_logistic_fit=apply_logistic_fit),
        "by_distortion_level": _group_metrics(pred_df, group_col="distortion_level", apply_logistic_fit=apply_logistic_fit),
        "by_content_name": _group_metrics(pred_df, group_col="content_name", apply_logistic_fit=False),
    }
    if save_analysis_path is not None:
        save_analysis_path = Path(save_analysis_path)
        save_analysis_path.parent.mkdir(parents=True, exist_ok=True)
        save_analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    return {**metrics, "analysis": analysis}
