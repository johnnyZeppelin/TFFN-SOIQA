from __future__ import annotations

from typing import Any

import torch


def apply_torch_runtime_settings(cfg: dict[str, Any]) -> None:
    project_cfg = cfg.get("project", {})
    num_threads = project_cfg.get("torch_num_threads", None)
    interop_threads = project_cfg.get("torch_num_interop_threads", None)

    if num_threads is not None:
        num_threads = int(num_threads)
        if num_threads > 0:
            torch.set_num_threads(num_threads)

    if interop_threads is not None:
        interop_threads = int(interop_threads)
        if interop_threads > 0:
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError:
                pass
