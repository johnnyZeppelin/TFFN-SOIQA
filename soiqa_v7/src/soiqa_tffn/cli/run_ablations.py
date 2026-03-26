from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from soiqa_tffn.config import load_config, save_yaml


ABLATIONS = {
    "full": {},
    "w_o_tpf_bd": {"model": {"use_tpf_bd": False}},
    "w_o_tpf_bs": {"model": {"use_tpf_bs": False}},
    "w_o_pdie": {"model": {"use_pdie": False}},
    "ff_simple_concat": {"model": {"ff_mode": "simple_concat"}},
    "double_side_swmsa": {"model": {"bd_double_side_swmsa": True}},
}


def _deep_update(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="configs/ablations")
    parser.add_argument("--ablation-output-root", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config, resolve_paths=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_ablation_root = str(Path(base_cfg["project"]["output_dir"]) / "ablations")
    ablation_output_root = Path(args.ablation_output_root or default_ablation_root)
    generated = []
    for name, update in ABLATIONS.items():
        cfg = _deep_update(base_cfg, update)
        cfg["project"]["name"] = f"{base_cfg['project']['name']}_{name}"
        cfg["project"]["output_dir"] = str(ablation_output_root / name)
        cfg["paths"]["train_manifest_path"] = str(Path(cfg["project"]["output_dir"]) / "train.csv")
        cfg["paths"]["test_manifest_path"] = str(Path(cfg["project"]["output_dir"]) / "test.csv")
        path = out_dir / f"{name}.yaml"
        save_yaml(cfg, path)
        generated.append(str(path))
    print(json.dumps({"generated": generated, "ablation_output_root": str(ablation_output_root)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
