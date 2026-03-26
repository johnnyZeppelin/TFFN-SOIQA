# SOIQA_TFFN

A package-first PyTorch implementation of a viewport-based SOIQA project inspired by the TFFN paper.

This version is updated specifically for the LIVE3DVR layout below and does **not** require you to change your dataset structure:

```text
/workspace/TFFN-SOIQA/data/
├─ Images/
├─ Images_r/
├─ view_ports/
│  └─ <stem>/
│     ├─ left/<stem>_left_01.png ... <stem>_left_20.png
│     └─ right/<stem>_right_01.png ... <stem>_right_20.png
├─ view_ports_r/
│  └─ <stem>/
│     ├─ left/<stem>_left_01_r.png ... <stem>_left_20_r.png
│     └─ right/<stem>_right_01_r.png ... <stem>_right_20_r.png
└─ DMOS.csv
```

The CSV is assumed to contain exactly these two columns:

- `ImageName`
- `DMOS`

## What is implemented now

- Package-safe project structure under `src/soiqa_tffn/`
- Config loader with automatic path resolution
- LIVE3DVR manifest generation directly from your current `left/` and `right/` subdirectory layout
- Metadata parsing into `content_name`, `distortion_type`, and `distortion_level`
- A dataset pipeline where **one sample = one SOI = 20 stereo viewport pairs**
- TFFN-style baseline model with TPF / PDIE / FF branches
- Train / eval / split / inspect / integrity-check / smoke-test entry scripts
- Repeated-run helper, ablation-config generator, and result-table summarizer
- PLCC / SRCC / KRCC / RMSE metrics with optional logistic fitting

## Important implementation note

The original paper is not open-sourced. This repository follows the paper structure where the details are clear and uses a practical, runnable approximation where details are missing. The PDIE branch is now wired to the official VMamba implementation. By default it looks for `/workspace/external_repos/VMamba/vmamba.py` and `/workspace/TFFN-SOIQA/vssms/vssm1_tiny_0230s_ckpt_epoch_264.pth`, and falls back to a vendored copy of the official `vmamba.py` when the external repo path is unavailable.

## Installation

```bash
cd SOIQA_TFFN
pip install -r requirements.txt
pip install -e .
```

Recommended for faster CUDA execution, following the upstream VMamba repo, but not strictly required because this project can fall back to the pure PyTorch scan path:

```bash
pip install triton
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

The wrapper scripts under `scripts/` can also be run directly from the project root without `pip install -e .`, because they auto-bootstrap the local `src/` path. Editable installation is still recommended.

## Current default config

`configs/default.yaml` is already pre-filled for your current LIVE3DVR path layout:

- `data_root: /workspace/TFFN-SOIQA/data`
- `csv_path: /workspace/TFFN-SOIQA/data/DMOS.csv`
- `live3dvr_dir: /workspace/TFFN-SOIQA/data/Images`
- `restored_live3dvr_dir: /workspace/TFFN-SOIQA/data/Images_r`
- `viewport_dir: /workspace/TFFN-SOIQA/data/view_ports`
- `restored_viewport_dir: /workspace/TFFN-SOIQA/data/view_ports_r`
- `dataset.stereo_packing_mode: nested_left_right`
- `manifest.image_name_col: ImageName`
- `manifest.score_col: DMOS`

## Main workflow

### 1) Build the manifest

```bash
python scripts/build_manifest.py --config configs/default.yaml
```

Outputs:

- `data/manifests/live3dvr_manifest.csv`
- `data/manifests/live3dvr_manifest_issues.csv` if anything is missing

The manifest stores one row per SOI and, for each of the 20 viewports, records explicit paired paths for left and right images.

### 2) Split train/test

```bash
python scripts/split_manifest.py --config configs/default.yaml
```

Outputs:

- `data/manifests/live3dvr_train.csv`
- `data/manifests/live3dvr_test.csv`
- `data/manifests/live3dvr_train_summary.json`

The default split strategy is `by_content`, which is usually safer than a pure random split.

### 3) Inspect one sample

```bash
python scripts/inspect_one.py --config configs/default.yaml --index 0
```

Optional preview image:

```bash
python scripts/inspect_one.py --config configs/default.yaml --index 0 --save-preview outputs/sample_preview.png
```

This preview now works for your nested `left/` and `right/` directory structure too.

### 4) Integrity check

```bash
python scripts/check_integrity.py --config configs/default.yaml
```

### 5) Smoke test

```bash
python scripts/smoke_test.py --config configs/default.yaml
```

This builds a tiny fake dataset and checks manifest generation, dataloader construction, and a forward pass.

### 6) Train

```bash
python scripts/train.py --config configs/default.yaml
```

Resume training:

```bash
python scripts/train.py --config configs/default.yaml --resume outputs/live3dvr_baseline/checkpoints/best.pt
```

Main outputs:

- `outputs/live3dvr_baseline/resolved_config.yaml`
- `outputs/live3dvr_baseline/history.csv`
- `outputs/live3dvr_baseline/checkpoints/*.pt`
- `outputs/live3dvr_baseline/preds/epoch_xxx_test_predictions.csv`
- `outputs/live3dvr_baseline/preds/epoch_xxx_test_analysis.json`

### 7) Evaluate the best checkpoint

```bash
python scripts/eval.py   --config configs/default.yaml   --checkpoint outputs/live3dvr_baseline/checkpoints/best.pt
```

Outputs:

- `outputs/live3dvr_baseline/preds/test_predictions.csv`
- `outputs/live3dvr_baseline/preds/eval_metrics.json`
- `outputs/live3dvr_baseline/preds/eval_analysis.json`

### 8) Generate ablation configs with isolated output folders

```bash
python scripts/run_ablations.py --config configs/default.yaml
```

This now generates configs that already point to separate output directories such as:

- `outputs/live3dvr_baseline/ablations/full`
- `outputs/live3dvr_baseline/ablations/w_o_tpf_bd`
- `outputs/live3dvr_baseline/ablations/w_o_tpf_bs`
- `outputs/live3dvr_baseline/ablations/w_o_pdie`
- `outputs/live3dvr_baseline/ablations/ff_simple_concat`
- `outputs/live3dvr_baseline/ablations/double_side_swmsa`

### 9) Run repeated splits

```bash
python scripts/run_repeats.py --config configs/default.yaml --num-runs 5
```

Outputs:

- `outputs/live3dvr_baseline/repeats/run_01/...`
- `outputs/live3dvr_baseline/repeats/run_02/...`
- ...
- `outputs/live3dvr_baseline/repeats/repeat_results.csv`
- `outputs/live3dvr_baseline/repeats/repeat_summary.json`

### 10) Export tables for papers or reports

Single-run tables:

```bash
python scripts/summarize_results.py   --single-run-dir outputs/live3dvr_baseline   --output-dir outputs/live3dvr_baseline/tables
```

Ablation table:

```bash
python scripts/summarize_results.py   --ablation-root outputs/live3dvr_baseline/ablations   --output-dir outputs/live3dvr_baseline/tables
```

Repeated-run tables:

```bash
python scripts/summarize_results.py   --repeat-root outputs/live3dvr_baseline/repeats   --output-dir outputs/live3dvr_baseline/tables
```

This writes CSV tables such as:

- `overall_metrics_table.csv`
- `by_distortion_type_table.csv`
- `by_distortion_level_table.csv`
- `by_content_name_table.csv`
- `ablation_table.csv`
- `repeat_results_table.csv`
- `repeat_summary_table.csv`

## Project layout

```text
SOIQA_TFFN/
├─ configs/
├─ data/
│  └─ manifests/
├─ outputs/
├─ scripts/
├─ src/
│  └─ soiqa_tffn/
│     ├─ cli/
│     ├─ data/
│     ├─ engine/
│     ├─ losses/
│     ├─ metrics/
│     ├─ models/
│     └─ utils/
├─ pyproject.toml
└─ requirements.txt
```
