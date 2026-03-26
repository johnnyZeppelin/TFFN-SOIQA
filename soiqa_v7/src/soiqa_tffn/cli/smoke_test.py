from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from soiqa_tffn.config import load_config
from soiqa_tffn.data import build_dataloader
from soiqa_tffn.data.manifest import build_live3dvr_manifest
from soiqa_tffn.models import build_model
from soiqa_tffn.utils import apply_torch_runtime_settings


def _make_fake_stereo_viewport(seed: int, input_size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    top = (rng.random((input_size, input_size, 3)) * 255).astype(np.uint8)
    bottom = (rng.random((input_size, input_size, 3)) * 255).astype(np.uint8)
    packed = np.concatenate([top, bottom], axis=0)
    return Image.fromarray(packed)


def _build_fake_dataset(root: Path, num_samples: int = 1, num_viewports: int = 1, input_size: int = 32) -> Path:
    (root / "LIVE3DVR").mkdir(parents=True, exist_ok=True)
    (root / "view_ports").mkdir(parents=True, exist_ok=True)
    (root / "restored" / "view_ports_restored").mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_idx in range(num_samples):
        stem = f"fake_{sample_idx:03d}"
        image_name = f"{stem}.png"
        Image.fromarray(np.zeros((512, 256, 3), dtype=np.uint8)).save(root / "LIVE3DVR" / image_name)
        for fov in range(1, num_viewports + 1):
            distorted = _make_fake_stereo_viewport(seed=sample_idx * 100 + fov, input_size=input_size)
            restored = _make_fake_stereo_viewport(seed=sample_idx * 100 + fov + 999, input_size=input_size)
            distorted.save(root / "view_ports" / f"{stem}_fov{fov}.png")
            restored.save(root / "restored" / "view_ports_restored" / f"{stem}_fov{fov}_r.png")
        rows.append({"Image Name": image_name, "DMOS": float(sample_idx)})
    csv_path = root / "dmos.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    apply_torch_runtime_settings(base_cfg)
    tmp_dir = Path(tempfile.mkdtemp(prefix="soiqa_tffn_smoke_"))
    try:
        _build_fake_dataset(tmp_dir, num_samples=1, num_viewports=1, input_size=32)
        cfg = base_cfg
        cfg["paths"]["data_root"] = str(tmp_dir)
        cfg["paths"]["csv_path"] = str(tmp_dir / "dmos.csv")
        cfg["paths"]["live3dvr_dir"] = str(tmp_dir / "LIVE3DVR")
        cfg["paths"]["viewport_dir"] = str(tmp_dir / "view_ports")
        cfg["paths"]["restored_root"] = str(tmp_dir / "restored")
        cfg["paths"]["restored_viewport_dir"] = str(tmp_dir / "restored" / "view_ports_restored")
        cfg["paths"]["manifest_path"] = str(tmp_dir / "manifest.csv")
        cfg["paths"]["train_manifest_path"] = str(tmp_dir / "train.csv")
        cfg["paths"]["test_manifest_path"] = str(tmp_dir / "test.csv")
        cfg["project"]["output_dir"] = str(tmp_dir / "outputs")
        cfg["project"]["num_workers"] = 0
        cfg["project"]["pin_memory"] = False
        cfg["manifest"]["num_viewports"] = 1
        cfg["manifest"]["image_name_col"] = "Image Name"
        cfg["dataset"]["stereo_packing_mode"] = "top_bottom"
        cfg["dataset"]["num_viewports"] = 1
        cfg["dataset"]["input_size"] = 32
        cfg["train"]["batch_size"] = 1
        cfg["eval"]["batch_size"] = 1
        cfg["model"]["stem_channels"] = 8
        cfg["model"]["layer1_channels"] = 8
        cfg["model"]["blocks_per_stage"] = 1
        cfg["model"]["stage_channels"] = [8, 16, 32]
        cfg["model"]["proj_dim"] = 4
        cfg["model"]["bd_dim"] = 4
        cfg["model"]["bs_dim"] = 4
        cfg["model"]["mf_dim"] = 4
        cfg["model"]["ff_hidden_dim"] = 8
        cfg["model"]["num_heads"] = 1
        cfg["model"]["pdie_depth"] = 1
        cfg["model"]["window_size"] = 2
        cfg["model"]["pdie_encoder"] = "vmamba_like"

        build_live3dvr_manifest(cfg)
        manifest_df = pd.read_csv(cfg["paths"]["manifest_path"])
        manifest_df.to_csv(cfg["paths"]["train_manifest_path"], index=False)
        manifest_df.to_csv(cfg["paths"]["test_manifest_path"], index=False)

        train_loader = build_dataloader(cfg, cfg["paths"]["train_manifest_path"], is_train=True)
        batch = next(iter(train_loader))
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            pred = model(batch["left_viewports"], batch["right_viewports"], batch["right_restored_viewports"])
        print("Smoke test passed.")
        print(f"Manifest rows: {len(manifest_df)}")
        print(f"Batch left_viewports: {tuple(batch['left_viewports'].shape)}")
        print(f"Pred shape: {tuple(pred.shape)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
