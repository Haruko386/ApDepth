# Last modified: 2024-05-24
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from apdepth import ApDepthPipeline
from src.util.seeding import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)


def get_prediction_path(output_dir, rgb_relative_path, name_mode):
    """Return the prediction path corresponding to one dataset RGB path."""
    rgb_basename = os.path.basename(rgb_relative_path)
    scene_dir = os.path.join(output_dir, os.path.dirname(rgb_relative_path))
    pred_basename = get_pred_name(rgb_basename, name_mode, suffix=".npy")
    return os.path.join(scene_dir, pred_basename)


def is_valid_prediction(prediction_path):
    """Check that a prediction exists and is a readable, non-empty NPY file."""
    if not os.path.isfile(prediction_path):
        return False

    try:
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        is_valid = prediction.size > 0
        mmap = getattr(prediction, "_mmap", None)
        if mmap is not None:
            mmap.close()
        return is_valid
    except (EOFError, OSError, ValueError):
        logging.warning(
            f"Incomplete or invalid prediction will be recomputed: "
            f"'{prediction_path}'"
        )
        return False


def get_pending_indices(dataset, output_dir):
    """Return unfinished dataset indices while preserving dataset-list order."""
    pending_indices = []
    for index, filename_line in enumerate(dataset.filenames):
        prediction_path = get_prediction_path(
            output_dir, filename_line[0], dataset.name_mode
        )
        if not is_valid_prediction(prediction_path):
            pending_indices.append(index)
    return pending_indices


def save_prediction_atomic(save_to, depth_pred):
    """Save without exposing a partially written final NPY file."""
    temporary_path = f"{save_to}.tmp"
    with open(temporary_path, "wb") as temporary_file:
        np.save(temporary_file, depth_pred)
    os.replace(temporary_path, save_to)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using ApDepth."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/apdepth",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/root/Dataset",
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--resume_run",
        action="store_true",
        help=(
            "Resume inference from an existing output directory. Valid prediction "
            "files are skipped and missing or incomplete files are recomputed."
        ),
    )

    # inference setting
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp32",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir
    resume_run = args.resume_run

    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    if resume_run:
        logging.info(
            "Resume mode enabled; scanning the output directory for completed "
            "predictions."
        )
    else:
        check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )

    inference_dataset = dataset
    if resume_run:
        pending_indices = get_pending_indices(dataset, output_dir)

        completed_count = len(dataset) - len(pending_indices)
        logging.info(
            f"Resume scan found {completed_count}/{len(dataset)} completed "
            f"predictions; {len(pending_indices)} remaining."
        )

        if not pending_indices:
            logging.info("All predictions are complete. Nothing to infer.")
            exit(0)

        first_pending_index = pending_indices[0]
        first_pending_rgb = dataset.filenames[first_pending_index][0]
        logging.info(
            f"First pending sample in dataset order: "
            f"{first_pending_index + 1}/{len(dataset)} '{first_pending_rgb}'"
        )
        inference_dataset = Subset(dataset, pending_indices)

    dataloader = DataLoader(inference_dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe = ApDepthPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
    # unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, f'unet'))
    # pipe.unet = unet
        

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            # Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)

            # Predict depth
            pipe_out = pipe(
                input_image,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            save_to = get_prediction_path(
                output_dir, rgb_filename, dataset.name_mode
            )
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            if os.path.exists(save_to):
                logging.warning(f"Existing file: '{save_to}' will be overwritten")

            save_prediction_atomic(save_to, depth_pred)
