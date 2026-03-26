from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageChops

from soiqa_tffn.config import load_config
from soiqa_tffn.data.datasets import Live3DVRDataset
from soiqa_tffn.data.stereo_viewport_parser import StereoParserConfig, StereoViewportParser


def _make_parser(cfg: dict) -> StereoViewportParser:
    return StereoViewportParser(
        StereoParserConfig(
            packing_mode=cfg["dataset"]["stereo_packing_mode"],
            input_size=int(cfg["dataset"]["input_size"]),
            resize_split_halves_back=bool(cfg["dataset"].get("resize_split_halves_back", True)),
            separate_file_suffix_left=cfg["dataset"].get("separate_file_suffix_left", "_L"),
            separate_file_suffix_right=cfg["dataset"].get("separate_file_suffix_right", "_R"),
            top_bottom_order=cfg["dataset"].get("top_bottom_order", "top_left_bottom_right"),
            left_right_order=cfg["dataset"].get("left_right_order", "left_left_right_right"),
        )
    )


def _open_source_preview(entry: str | dict[str, str]) -> Image.Image:
    if isinstance(entry, str):
        return Image.open(entry).convert("RGB")
    left = Image.open(entry["left"]).convert("RGB")
    right = Image.open(entry["right"]).convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), color=(255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _save_preview(distorted_entry: str | dict[str, str], restored_entry: str | dict[str, str], parser: StereoViewportParser, output_path: Path) -> None:
    src = _open_source_preview(distorted_entry)
    restored_src = _open_source_preview(restored_entry)
    left, right = parser(distorted_entry)
    _, restored_right = parser(restored_entry)
    diff = ImageOps.autocontrast(ImageChops.difference(right, restored_right))

    tiles = [src, left, right, restored_src, restored_right, diff]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 2), color=(255, 255, 255))
    for idx, tile in enumerate(tiles):
        row = idx // 3
        col = idx % 3
        canvas.paste(tile.resize((tile_w, tile_h)), (col * tile_w, row * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--save-preview", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = args.manifest or cfg["paths"]["train_manifest_path"]
    dataset = Live3DVRDataset(manifest_path=manifest_path, cfg=cfg, is_train=False)
    sample = dataset[args.index]
    row = dataset.df.iloc[args.index]

    print(f"image_name: {sample['image_name']}")
    for key in ["content_name", "distortion_type", "distortion_level", "stereo_packing_mode"]:
        if key in row.index:
            print(f"{key}: {row[key]}")
    print(f"left_viewports: {tuple(sample['left_viewports'].shape)}")
    print(f"right_viewports: {tuple(sample['right_viewports'].shape)}")
    print(f"right_restored_viewports: {tuple(sample['right_restored_viewports'].shape)}")
    if "left_restored_viewports" in sample:
        print(f"left_restored_viewports: {tuple(sample['left_restored_viewports'].shape)}")
    print(f"score: {float(sample['score'])}")

    if args.save_preview:
        distorted_entries: list[Any] = json.loads(row["distorted_viewports_json"])
        restored_entries: list[Any] = json.loads(row["restored_viewports_json"])
        parser_obj = _make_parser(cfg)
        _save_preview(distorted_entries[0], restored_entries[0], parser_obj, Path(args.save_preview))
        print(f"Saved preview to: {args.save_preview}")


if __name__ == "__main__":
    main()
