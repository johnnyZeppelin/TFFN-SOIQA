from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_ORDER = ["PLCC", "SRCC", "KRCC", "RMSE", "PLCC_raw", "RMSE_raw"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _analysis_table(analysis: dict[str, Any], key: str) -> pd.DataFrame:
    group = analysis.get(key, {})
    rows = []
    for name, metrics in group.items():
        row = {"group": name, **{k: _maybe_float(metrics.get(k)) for k in ["count", *METRIC_ORDER] if k in metrics}}
        rows.append(row)
    return pd.DataFrame(rows)


def _single_run_tables(run_dir: Path, output_dir: Path) -> list[Path]:
    pred_dir = run_dir / "preds"
    metrics_path = pred_dir / "eval_metrics.json"
    analysis_path = pred_dir / "eval_analysis.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing eval metrics: {metrics_path}")
    metrics = _read_json(metrics_path)
    tables: list[Path] = []
    overall_df = pd.DataFrame([{"run_name": run_dir.name, **{k: _maybe_float(metrics.get(k)) for k in METRIC_ORDER if k in metrics}}])
    overall_path = output_dir / "overall_metrics_table.csv"
    overall_df.to_csv(overall_path, index=False)
    tables.append(overall_path)
    if analysis_path.exists():
        analysis = _read_json(analysis_path)
        for key, name in [
            ("by_distortion_type", "by_distortion_type_table.csv"),
            ("by_distortion_level", "by_distortion_level_table.csv"),
            ("by_content_name", "by_content_name_table.csv"),
        ]:
            df = _analysis_table(analysis, key)
            if not df.empty:
                out = output_dir / name
                df.to_csv(out, index=False)
                tables.append(out)
    return tables


def _collect_ablation_rows(root_dir: Path) -> pd.DataFrame:
    rows = []
    for eval_metrics in sorted(root_dir.glob("*/preds/eval_metrics.json")):
        variant = eval_metrics.parent.parent.name
        metrics = _read_json(eval_metrics)
        row = {"variant": variant, **{k: _maybe_float(metrics.get(k)) for k in METRIC_ORDER if k in metrics}}
        rows.append(row)
    return pd.DataFrame(rows)


def _repeat_tables(repeat_root: Path, output_dir: Path) -> list[Path]:
    tables: list[Path] = []
    results_csv = repeat_root / "repeat_results.csv"
    summary_json = repeat_root / "repeat_summary.json"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        out = output_dir / "repeat_results_table.csv"
        df.to_csv(out, index=False)
        tables.append(out)
    if summary_json.exists():
        summary = _read_json(summary_json)
        metrics = summary.get("metrics", {})
        rows = []
        for metric_name, values in metrics.items():
            rows.append({"metric": metric_name, "mean": _maybe_float(values.get("mean")), "std": _maybe_float(values.get("std"))})
        if rows:
            out = output_dir / "repeat_summary_table.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
            tables.append(out)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run-dir", type=str, default=None)
    parser.add_argument("--ablation-root", type=str, default=None)
    parser.add_argument("--repeat-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    if args.single_run_dir:
        generated.extend(str(p) for p in _single_run_tables(Path(args.single_run_dir), output_dir))
    if args.ablation_root:
        df = _collect_ablation_rows(Path(args.ablation_root))
        if not df.empty:
            ablation_path = output_dir / "ablation_table.csv"
            df.to_csv(ablation_path, index=False)
            generated.append(str(ablation_path))
    if args.repeat_root:
        generated.extend(str(p) for p in _repeat_tables(Path(args.repeat_root), output_dir))

    print(json.dumps({"generated": generated}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
