"""Inspect real training targets without loading SD2 or DA2 checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import DatasetMode, get_dataset
from src.util.config_util import recursive_load_config
from src.util.depth_transform import get_depth_normalizer
from src.util.far_supervision import prepare_far_supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_sky_finetune.yaml")
    parser.add_argument("--base_data_dir", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output_dir", default="output/far_audit")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    cfg = recursive_load_config(args.config)
    configs = cfg.dataset.train.dataset_list if cfg.dataset.train.name == "mixed" else [cfg.dataset.train]
    vkitti = [c for c in configs if c.name == "vkitti"]
    if not vkitti:
        parser.error("No VKITTI dataset found in training config")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_index, data_cfg in enumerate(vkitti):
        dataset = get_dataset(data_cfg, args.base_data_dir, DatasetMode.TRAIN,
                              depth_transform=get_depth_normalizer(cfg.depth_normalization))
        if not len(dataset):
            raise RuntimeError("VKITTI training split is empty")
        indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
        for index in indices:
            item = dataset[int(index)]
            far = item["known_far_mask"]
            target, _, added, _ = prepare_far_supervision(
                item[cfg.gt_depth_type][None], item[cfg.gt_mask_type][None], far[None])
            record = {
                "rgb": item["rgb_relative_path"],
                "known_far_pixel_ratio": far.float().mean().item(),
                "added_latent_ratio": added.float().mean().item(),
                "saturated_pixel_ratio": (item["depth_raw_linear"] >= 655.3).float().mean().item(),
            }
            records.append(record)
            prefix = destination / f"{dataset_index}_{index:06d}"
            Image.fromarray(item["rgb_int"].permute(1, 2, 0).numpy().astype(np.uint8)).save(f"{prefix}_rgb.png")
            Image.fromarray(far[0].numpy().astype(np.uint8) * 255).save(f"{prefix}_far_mask.png")
            gray = ((target[0, 0].clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
            Image.fromarray(gray).save(f"{prefix}_target_far_white.png")
            print(json.dumps(record))
    (destination / "coverage.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if not any(r["added_latent_ratio"] > 0 for r in records):
        raise RuntimeError("No far training support found. Inspect native depth encoding before training.")


if __name__ == "__main__":
    main()
