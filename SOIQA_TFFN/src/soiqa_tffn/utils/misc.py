from __future__ import annotations

from pathlib import Path


def stem_from_image_name(image_name: str) -> str:
    return Path(image_name).stem
