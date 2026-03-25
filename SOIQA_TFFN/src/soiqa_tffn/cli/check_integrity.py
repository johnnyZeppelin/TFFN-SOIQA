from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soiqa_tffn.config import load_config
from soiqa_tffn.data.manifest import validate_manifest_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest or cfg["paths"]["manifest_path"])
    df = pd.read_csv(manifest_path)
    issues = validate_manifest_dataframe(df, expected_viewports=int(cfg["manifest"]["num_viewports"]))
    if not issues:
        print(f"Integrity check passed: {manifest_path}")
        return
    print(f"Integrity check found {len(issues)} issues in {manifest_path}:")
    for issue in issues[:50]:
        print(f"- {issue.stem} | {issue.kind} | {issue.detail}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
