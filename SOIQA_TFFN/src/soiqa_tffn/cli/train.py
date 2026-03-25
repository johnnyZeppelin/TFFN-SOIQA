from __future__ import annotations

import argparse
from pathlib import Path

from soiqa_tffn.config import load_config
from soiqa_tffn.data import build_dataloader
from soiqa_tffn.data.score_normalizer import ScoreNormalizer
from soiqa_tffn.engine import load_checkpoint, train_model
from soiqa_tffn.losses import build_loss
from soiqa_tffn.models import build_model
from soiqa_tffn.utils import apply_torch_runtime_settings, seed_everything, setup_logger


def _build_score_normalizer(cfg: dict, train_loader) -> ScoreNormalizer:
    norm_cfg = cfg.get("score_norm", {})
    enabled = bool(norm_cfg.get("enabled", False))
    mode = str(norm_cfg.get("mode", "none")) if enabled else "none"
    if mode == "none":
        return ScoreNormalizer(mode="none")
    values: list[float] = []
    for item in train_loader.dataset.df["dmos"].tolist():
        values.append(float(item))
    return ScoreNormalizer.fit(values, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--test-manifest", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_torch_runtime_settings(cfg)
    output_dir = Path(cfg["project"]["output_dir"])
    logger = setup_logger(cfg["project"]["name"], save_dir=output_dir / "logs")
    seed_everything(int(cfg["project"]["seed"]))

    train_manifest_path = args.train_manifest or cfg["paths"]["train_manifest_path"]
    test_manifest_path = args.test_manifest or cfg["paths"]["test_manifest_path"]

    train_loader = build_dataloader(cfg, train_manifest_path, is_train=True)
    test_loader = build_dataloader(cfg, test_manifest_path, is_train=False)
    model = build_model(cfg)
    criterion = build_loss(cfg["loss"]["name"])
    score_normalizer = _build_score_normalizer(cfg, train_loader)

    logger.info("Start training.")
    logger.info(f"Train manifest: {train_manifest_path}")
    logger.info(f"Test manifest: {test_manifest_path}")
    logger.info(f"Train size: {len(train_loader.dataset)} | Test size: {len(test_loader.dataset)}")
    logger.info(
        "Model switches | "
        f"use_tpf_bd={cfg['model'].get('use_tpf_bd', True)} | "
        f"use_tpf_bs={cfg['model'].get('use_tpf_bs', True)} | "
        f"use_pdie={cfg['model'].get('use_pdie', True)} | "
        f"ff_mode={cfg['model'].get('ff_mode', 'ff')} | "
        f"bd_double_side_swmsa={cfg['model'].get('bd_double_side_swmsa', False)}"
    )

    resume_state = None
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        resume_state = load_checkpoint(args.resume, map_location="cpu")

    train_model(
        cfg,
        model,
        train_loader,
        test_loader,
        criterion,
        logger,
        score_normalizer=score_normalizer,
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main()
