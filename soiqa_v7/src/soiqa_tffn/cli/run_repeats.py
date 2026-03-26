from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd

from soiqa_tffn.config import load_config, save_yaml
from soiqa_tffn.data import build_dataloader
from soiqa_tffn.data.score_normalizer import ScoreNormalizer
from soiqa_tffn.data.split import save_split_summary, split_manifest_dataframe, summarize_split
from soiqa_tffn.engine import train_model
from soiqa_tffn.losses import build_loss
from soiqa_tffn.models import build_model
from soiqa_tffn.utils import apply_torch_runtime_settings, seed_everything, setup_logger


def _build_score_normalizer(cfg: dict, train_loader) -> ScoreNormalizer:
    norm_cfg = cfg.get("score_norm", {})
    enabled = bool(norm_cfg.get("enabled", False))
    mode = str(norm_cfg.get("mode", "none")) if enabled else "none"
    if mode == "none":
        return ScoreNormalizer(mode="none")
    values = [float(v) for v in train_loader.dataset.df["dmos"].tolist()]
    return ScoreNormalizer.fit(values, mode=mode)


def _metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    vals = [float(r[metric]) for r in rows if metric in r]
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(mean(vals)), "std": float(pstdev(vals) if len(vals) > 1 else 0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    apply_torch_runtime_settings(base_cfg)
    manifest_path = Path(args.manifest or base_cfg["paths"]["manifest_path"])
    output_root = Path(args.output_root or (Path(base_cfg["project"]["output_dir"]) / "repeats"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_csv(manifest_path)
    seed_base = int(args.seed_base if args.seed_base is not None else base_cfg["split"].get("seed", 3407))

    all_rows: list[dict[str, Any]] = []
    for run_idx in range(args.num_runs):
        run_seed = seed_base + run_idx
        cfg = load_config(args.config)
        apply_torch_runtime_settings(cfg)
        run_dir = output_root / f"run_{run_idx + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["project"]["output_dir"] = str(run_dir)
        cfg["project"]["seed"] = run_seed
        cfg["split"]["seed"] = run_seed
        cfg["paths"]["train_manifest_path"] = str(run_dir / "train.csv")
        cfg["paths"]["test_manifest_path"] = str(run_dir / "test.csv")
        save_yaml(cfg, run_dir / "resolved_repeat_config.yaml")

        split_result = split_manifest_dataframe(cfg, manifest_df, seed=run_seed)
        split_result.train_df.to_csv(cfg["paths"]["train_manifest_path"], index=False)
        split_result.test_df.to_csv(cfg["paths"]["test_manifest_path"], index=False)
        save_split_summary(summarize_split(split_result.train_df, split_result.test_df), run_dir / "split_summary.json")

        logger = setup_logger(f"{cfg['project']['name']}_run{run_idx + 1:02d}", save_dir=run_dir / "logs")
        seed_everything(run_seed)
        train_loader = build_dataloader(cfg, cfg["paths"]["train_manifest_path"], is_train=True)
        test_loader = build_dataloader(cfg, cfg["paths"]["test_manifest_path"], is_train=False)
        model = build_model(cfg)
        criterion = build_loss(cfg["loss"]["name"])
        normalizer = _build_score_normalizer(cfg, train_loader)
        best_metrics = train_model(cfg, model, train_loader, test_loader, criterion, logger, score_normalizer=normalizer)
        row = {"run_idx": run_idx + 1, "seed": run_seed, **best_metrics}
        all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_root / "repeat_results.csv", index=False)
    summary = {
        "num_runs": int(args.num_runs),
        "metrics": {m: _metric_summary(all_rows, m) for m in ["PLCC", "SRCC", "KRCC", "RMSE"]},
    }
    (output_root / "repeat_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(results_df)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
