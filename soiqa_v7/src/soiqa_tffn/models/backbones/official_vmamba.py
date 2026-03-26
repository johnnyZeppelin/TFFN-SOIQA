from __future__ import annotations

import os
import re
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from soiqa_tffn.models.norms import SafeLayerNorm

_TORCHVISION_NMS_STUB = None


def _ensure_torchvision_nms_stub() -> None:
    global _TORCHVISION_NMS_STUB
    if _TORCHVISION_NMS_STUB is not None:
        return
    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_NMS_STUB = lib
    except Exception:
        _TORCHVISION_NMS_STUB = False


def _preprocess_vmamba_source(source: str, disable_triton: bool) -> str:
    if not disable_triton:
        return source

    start_token = "WITH_TRITON = True"
    end_token = "# torch implementation ======================================="
    start = source.find(start_token)
    end = source.find(end_token)
    if start < 0 or end < 0 or end <= start:
        return source

    replacement = """WITH_TRITON = False

class _FakeTriton:
    @staticmethod
    def jit(fn=None, **kwargs):
        if fn is None:
            return lambda real_fn: real_fn
        return fn

    @staticmethod
    def cdiv(x, y):
        return (x + y - 1) // y


class _FakeTL:
    constexpr = int


triton = _FakeTriton()
tl = _FakeTL()

"""
    return source[:start] + replacement + source[end:]


def _load_module_from_file(module_name: str, file_path: Path, disable_triton: bool = False):
    source = file_path.read_text(encoding="utf-8")
    source = _preprocess_vmamba_source(source, disable_triton=disable_triton)
    module = types.ModuleType(module_name)
    module.__file__ = str(file_path)
    sys.modules[module_name] = module
    exec(compile(source, str(file_path), "exec"), module.__dict__)
    return module
def _resolve_vmamba_module(source: str = "auto", repo_root: str | os.PathLike[str] | None = None, disable_triton: bool = False):
    _ensure_torchvision_nms_stub()
    source = str(source).lower()
    external_file = None
    if repo_root is not None:
        external_file = Path(repo_root).expanduser().resolve() / "vmamba.py"

    if source not in {"auto", "external", "vendor"}:
        raise ValueError(f"Unsupported VMamba source: {source}")

    if source in {"auto", "external"} and external_file is not None and external_file.exists():
        return _load_module_from_file("soiqa_tffn_external_vmamba", external_file, disable_triton=disable_triton)
    if source == "external":
        raise FileNotFoundError(f"Configured VMamba repo root does not contain vmamba.py: {external_file}")

    vendor_file = Path(__file__).resolve().parent / "vendor_vmamba.py"
    return _load_module_from_file("soiqa_tffn_vendor_vmamba", vendor_file, disable_triton=disable_triton)


OFFICIAL_VMAMBA_VARIANTS: dict[str, dict[str, Any]] = {
    "vmamba_tiny_s1l8": {
        "depths": [2, 2, 8, 2],
        "dims": 96,
        "drop_path_rate": 0.2,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": 1000,
        "ssm_d_state": 1,
        "ssm_ratio": 1.0,
        "ssm_dt_rank": "auto",
        "ssm_act_layer": "silu",
        "ssm_conv": 3,
        "ssm_conv_bias": False,
        "ssm_drop_rate": 0.0,
        "ssm_init": "v0",
        "forward_type": "v05_noz",
        "mlp_ratio": 4.0,
        "mlp_act_layer": "gelu",
        "mlp_drop_rate": 0.0,
        "gmlp": False,
        "patch_norm": True,
        "downsample_version": "v3",
        "patchembed_version": "v2",
        "use_checkpoint": False,
        "posembed": False,
        "imgsize": 224,
    },
    "vmamba_small_s1l20": {
        "depths": [2, 2, 20, 2],
        "dims": 96,
        "drop_path_rate": 0.3,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": 1000,
        "ssm_d_state": 1,
        "ssm_ratio": 1.0,
        "ssm_dt_rank": "auto",
        "ssm_act_layer": "silu",
        "ssm_conv": 3,
        "ssm_conv_bias": False,
        "ssm_drop_rate": 0.0,
        "ssm_init": "v0",
        "forward_type": "v05_noz",
        "mlp_ratio": 4.0,
        "mlp_act_layer": "gelu",
        "mlp_drop_rate": 0.0,
        "gmlp": False,
        "patch_norm": True,
        "downsample_version": "v3",
        "patchembed_version": "v2",
        "use_checkpoint": False,
        "posembed": False,
        "imgsize": 224,
    },
    "vmamba_base_s1l20": {
        "depths": [2, 2, 20, 2],
        "dims": 128,
        "drop_path_rate": 0.5,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": 1000,
        "ssm_d_state": 1,
        "ssm_ratio": 1.0,
        "ssm_dt_rank": "auto",
        "ssm_act_layer": "silu",
        "ssm_conv": 3,
        "ssm_conv_bias": False,
        "ssm_drop_rate": 0.0,
        "ssm_init": "v0",
        "forward_type": "v05_noz",
        "mlp_ratio": 4.0,
        "mlp_act_layer": "gelu",
        "mlp_drop_rate": 0.0,
        "gmlp": False,
        "patch_norm": True,
        "downsample_version": "v3",
        "patchembed_version": "v2",
        "use_checkpoint": False,
        "posembed": False,
        "imgsize": 224,
    },
    "vmamba_tiny_s2l5": {
        "depths": [2, 2, 5, 2],
        "dims": 96,
        "drop_path_rate": 0.2,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": 1000,
        "ssm_d_state": 1,
        "ssm_ratio": 2.0,
        "ssm_dt_rank": "auto",
        "ssm_act_layer": "silu",
        "ssm_conv": 3,
        "ssm_conv_bias": False,
        "ssm_drop_rate": 0.0,
        "ssm_init": "v0",
        "forward_type": "v05_noz",
        "mlp_ratio": 4.0,
        "mlp_act_layer": "gelu",
        "mlp_drop_rate": 0.0,
        "gmlp": False,
        "patch_norm": True,
        "downsample_version": "v3",
        "patchembed_version": "v2",
        "use_checkpoint": False,
        "posembed": False,
        "imgsize": 224,
    },
}


def get_variant_stage_dims(variant_cfg: dict[str, Any]) -> list[int]:
    dims = variant_cfg["dims"]
    num_layers = len(variant_cfg["depths"])
    if isinstance(dims, int):
        return [int(dims * (2**i)) for i in range(num_layers)]
    return [int(x) for x in dims]


class OfficialVMambaEncoder(nn.Module):
    def __init__(
        self,
        mf_dim: int,
        num_viewports: int = 20,
        concat_mode: str = "grid",
        variant: str = "vmamba_tiny_s1l8",
        source: str = "auto",
        repo_root: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        pretrained: bool = True,
        strict_load: bool = False,
        out_index: int = 3,
        input_size: tuple[int, int] = (224, 224),
        grid_layout: tuple[int, int] | None = (4, 5),
        channel_first: bool = True,
        disable_triton: bool = True,
        force_torch_scan: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_viewports = int(num_viewports)
        self.concat_mode = str(concat_mode).lower()
        if self.concat_mode not in {"grid", "channel"}:
            raise ValueError(f"Unsupported official VMamba concat mode: {self.concat_mode}")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.grid_layout = tuple(grid_layout) if grid_layout is not None else None
        self.variant = str(variant)
        if self.variant not in OFFICIAL_VMAMBA_VARIANTS:
            raise KeyError(
                f"Unsupported VMamba variant '{self.variant}'. Supported: {sorted(OFFICIAL_VMAMBA_VARIANTS)}"
            )

        self.vmamba = _resolve_vmamba_module(source=source, repo_root=repo_root, disable_triton=disable_triton)
        if disable_triton and hasattr(self.vmamba, "WITH_TRITON"):
            self.vmamba.WITH_TRITON = False
        if force_torch_scan and hasattr(self.vmamba, "WITH_SELECTIVESCAN_MAMBA"):
            self.vmamba.WITH_SELECTIVESCAN_MAMBA = False

        variant_cfg = dict(OFFICIAL_VMAMBA_VARIANTS[self.variant])
        variant_cfg["norm_layer"] = "ln2d" if channel_first else "ln"
        variant_cfg["imgsize"] = max(self.input_size)
        self.stage_dims = get_variant_stage_dims(variant_cfg)
        self.out_index = int(out_index)
        if not (0 <= self.out_index < len(self.stage_dims)):
            raise ValueError(f"Invalid VMamba out_index {self.out_index} for stage dims {self.stage_dims}")
        self.feature_dim = self.stage_dims[self.out_index]

        self.input_adapter: nn.Module
        if self.concat_mode == "channel":
            in_channels = 3 * self.num_viewports
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 3, kernel_size=1, bias=True),
            )
        else:
            self.input_adapter = nn.Identity()

        self.backbone = self.vmamba.Backbone_VSSM(
            out_indices=(self.out_index,),
            pretrained=None,
            **variant_cfg,
        )
        if pretrained:
            self._load_checkpoint_flexible(checkpoint_path=checkpoint_path, strict=strict_load)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, mf_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = SafeLayerNorm(mf_dim)

    def _resolve_grid_layout(self) -> tuple[int, int]:
        if self.grid_layout is not None:
            gh, gw = int(self.grid_layout[0]), int(self.grid_layout[1])
            if gh * gw != self.num_viewports:
                raise ValueError(
                    f"Configured grid_layout {self.grid_layout} is incompatible with {self.num_viewports} viewports."
                )
            return gh, gw
        gh = int(round(self.num_viewports ** 0.5))
        while gh > 1 and (self.num_viewports % gh) != 0:
            gh -= 1
        gw = self.num_viewports // gh
        return gh, gw

    def _concat_diff_maps(self, diff: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = diff.shape
        if n != self.num_viewports:
            raise ValueError(f"OfficialVMambaEncoder expected {self.num_viewports} viewports, but got {n}.")
        if self.concat_mode == "channel":
            return diff.reshape(b, n * c, h, w)

        grid_h, grid_w = self._resolve_grid_layout()
        diff = diff.view(b, grid_h, grid_w, c, h, w)
        diff = diff.permute(0, 3, 1, 4, 2, 5).contiguous()
        return diff.view(b, c, grid_h * h, grid_w * w)

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("model", "state_dict", "model_ema"):
                if key in payload and isinstance(payload[key], dict):
                    return payload[key]
            if all(isinstance(k, str) for k in payload.keys()):
                return payload
        raise RuntimeError("Unsupported checkpoint format for VMamba weight loading.")

    def _load_checkpoint_flexible(self, checkpoint_path: str | os.PathLike[str] | None, strict: bool = False) -> None:
        if checkpoint_path is None:
            warnings.warn("VMamba checkpoint_path is None. The official VMamba branch will use random initialization.")
            return
        ckpt_path = Path(checkpoint_path).expanduser()
        if not ckpt_path.exists():
            warnings.warn(
                f"VMamba checkpoint not found at {ckpt_path}. The official VMamba branch will use random initialization."
            )
            return
        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = self._extract_state_dict(payload)
        incompatible = self.backbone.load_state_dict(state_dict, strict=strict)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            warnings.warn(
                "Loaded VMamba checkpoint with non-strict matching. "
                f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._concat_diff_maps(x)
        x = self.input_adapter(x)
        if self.input_size is not None:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        feats = self.backbone(x)
        feat = feats[0] if isinstance(feats, (list, tuple)) else feats
        feat = self.pool(feat).flatten(1)
        feat = self.proj(feat)
        return self.norm(feat)
