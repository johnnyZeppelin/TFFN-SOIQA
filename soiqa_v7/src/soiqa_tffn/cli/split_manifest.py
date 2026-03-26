from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soiqa_tffn.config import load_config
from soiqa_tffn.data.split import save_split_summary, split_manifest_dataframe, summarize_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--train-out", type=str, default=None)
    parser.add_argument("--test-out", type=str, default=None)
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    manifest_path = Path(args.manifest or cfg["paths"]["manifest_path"])
    train_path = Path(args.train_out or cfg["paths"]["train_manifest_path"])
    test_path = Path(args.test_out or cfg["paths"]["test_manifest_path"])

    df = pd.read_csv(manifest_path)
    split_result = split_manifest_dataframe(
        cfg,
        df,
        seed=args.seed,
        strategy=args.strategy,
        train_ratio=args.train_ratio,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    split_result.train_df.to_csv(train_path, index=False)
    split_result.test_df.to_csv(test_path, index=False)

    summary = summarize_split(split_result.train_df, split_result.test_df)
    summary_path = Path(args.summary_out) if args.summary_out else (train_path.parent / f"{train_path.stem}_summary.json")
    save_split_summary(summary, summary_path)

    print(f"Saved train split to {train_path} ({len(split_result.train_df)} samples)")
    print(f"Saved test split to {test_path} ({len(split_result.test_df)} samples)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
