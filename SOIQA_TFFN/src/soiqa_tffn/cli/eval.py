from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from soiqa_tffn.config import load_config
from soiqa_tffn.data import build_dataloader
from soiqa_tffn.data.score_normalizer import ScoreNormalizer
from soiqa_tffn.engine import evaluate_model, load_checkpoint
from soiqa_tffn.models import build_model
from soiqa_tffn.utils import apply_torch_runtime_settings, setup_logger


def _select_device(cfg: dict) -> torch.device:
    requested_device = str(cfg["project"].get("device", "auto"))
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested_device)


def _load_score_normalizer(ckpt: dict, output_dir: Path) -> ScoreNormalizer:
    if isinstance(ckpt.get("score_normalizer"), dict):
        return ScoreNormalizer.from_dict(ckpt["score_normalizer"])
    json_path = output_dir / "score_normalizer.json"
    if json_path.exists():
        return ScoreNormalizer.load(json_path)
    return ScoreNormalizer(mode="none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--save-metrics-json", type=str, default=None)
    parser.add_argument("--save-predictions-csv", type=str, default=None)
    parser.add_argument("--save-analysis-json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_torch_runtime_settings(cfg)
    logger = setup_logger(cfg["project"]["name"], save_dir=Path(cfg["project"]["output_dir"]) / "logs")
    device = _select_device(cfg)

    model = build_model(cfg)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    manifest_path = args.manifest or cfg["paths"]["test_manifest_path"]
    dataloader = build_dataloader(cfg, manifest_path, is_train=False)
    output_dir = Path(cfg["project"]["output_dir"])
    pred_path = Path(args.save_predictions_csv) if args.save_predictions_csv else (output_dir / "preds" / cfg["eval"].get("prediction_csv_name", "test_predictions.csv"))
    analysis_path = Path(args.save_analysis_json) if args.save_analysis_json else (output_dir / "preds" / cfg["eval"].get("analysis_json_name", "eval_analysis.json"))
    score_normalizer = _load_score_normalizer(ckpt, output_dir=output_dir)
    metrics = evaluate_model(
        model,
        dataloader,
        device=device,
        save_predictions_path=pred_path,
        save_analysis_path=analysis_path,
        score_normalizer=score_normalizer,
        apply_logistic_fit=bool(cfg.get("eval", {}).get("apply_logistic_fit", False)),
    )
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Eval metrics: { {k: v for k, v in metrics.items() if k != 'analysis'} }")

    metrics_path = Path(args.save_metrics_json or (output_dir / "preds" / "eval_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
