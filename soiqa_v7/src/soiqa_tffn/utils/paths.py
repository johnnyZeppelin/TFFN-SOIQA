from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PATH_KEY_HINTS = (
    "path",
    "dir",
    "root",
    "file",
    "csv",
    "checkpoint",
    "manifest",
    "output",
)


def expand_path(path_like: str | Path) -> Path:
    return Path(os.path.expandvars(str(path_like))).expanduser()


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    path = expand_path(path_like)
    if path.is_absolute() or base_dir is None:
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def maybe_resolve_path_value(key: str, value: Any, base_dir: str | Path) -> Any:
    if not isinstance(value, str):
        return value
    lowered = key.lower()
    if lowered.endswith("_name") or lowered in {"image_name_col", "score_col", "group_col"}:
        return value
    if any(hint in lowered for hint in PATH_KEY_HINTS):
        return str(resolve_path(value, base_dir=base_dir))
    return value


def recursively_resolve_paths(data: Any, base_dir: str | Path) -> Any:
    if isinstance(data, dict):
        return {k: recursively_resolve_paths(maybe_resolve_path_value(k, v, base_dir), base_dir) for k, v in data.items()}
    if isinstance(data, list):
        return [recursively_resolve_paths(v, base_dir) for v in data]
    return data
