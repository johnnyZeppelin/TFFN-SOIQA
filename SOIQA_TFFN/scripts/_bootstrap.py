from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_local_src() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    src_str = str(src_dir)
    if src_dir.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)
