from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(value: str) -> Any:
    return json.loads(value)


def write_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
