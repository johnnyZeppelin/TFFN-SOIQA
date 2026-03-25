from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from soiqa_tffn.utils.paths import recursively_resolve_paths


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _inject_runtime_context(cfg: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    project_root = config_dir.parent if config_dir.name == "configs" else config_dir
    cfg = deepcopy(cfg)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["config_path"] = str(config_path)
    cfg["runtime"]["config_dir"] = str(config_dir)
    cfg["runtime"]["project_root"] = str(project_root)
    return cfg


def load_config(config_path: str | Path, extra_paths: list[str | Path] | None = None, resolve_paths: bool = True) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    for extra in extra_paths or []:
        cfg = _deep_update(cfg, load_yaml(extra))
    cfg = _inject_runtime_context(cfg, config_path)
    if resolve_paths:
        cfg = recursively_resolve_paths(cfg, base_dir=cfg["runtime"]["project_root"])
    return cfg
