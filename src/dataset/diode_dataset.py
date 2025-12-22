# Last modified: 2025-12-22
#
# Copyright 2025 Jiawei Wang SJZU. All rights reserved.
#
# This file has been modified from the original version.
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
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import os
import tarfile
from io import BytesIO

import numpy as np
import torch
from scipy import ndimage  # <--- Added for Sobel filter

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode


class DIODEDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # DIODE data parameter
            min_depth=0.6,
            max_depth=350,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_npy_file(self, rel_path):
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            fileobj = self.tar_obj.extractfile("./" + rel_path)
            npy_path_or_content = BytesIO(fileobj.read())
        else:
            npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
        data = np.load(npy_path_or_content).squeeze()[np.newaxis, :, :]
        return data

    def _read_depth_file(self, rel_path):
        depth = self._read_npy_file(rel_path)
        return depth

    def _get_data_path(self, index):
        return self.filenames[index]

    def _get_data_item(self, index):
        # Special: depth mask is read from data

        rgb_rel_path, depth_rel_path, mask_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=None
            )
            rasters.update(depth_data)

            # --- START: Modified Mask Logic with Sobel Filter ---
            
            # 1. Get raw depth as numpy array [H, W] for scipy processing
            # depth_raw_linear is [1, H, W] tensor
            depth_tensor = rasters["depth_raw_linear"]
            depth_numpy = depth_tensor.squeeze().cpu().numpy()

            # 2. Read original mask provided by dataset
            mask_numpy = self._read_npy_file(mask_rel_path).astype(bool).squeeze()

            # 3. Apply Sobel gradient filter (replicated from diode.py)
            # This removes high-frequency edges where prediction is often difficult/ambiguous
            dx = ndimage.sobel(depth_numpy, 0)  # horizontal derivative
            dy = ndimage.sobel(depth_numpy, 1)  # vertical derivative
            grad = np.abs(dx) + np.abs(dy)
            
            # Create edge mask: keep pixels where gradient is low
            edge_mask = grad <= 0.3

            # 4. Apply range and validity checks
            # Replicating logic: valid_mask & (depth >= 0.6) & (depth <= 350) & (~np.isnan) & (~np.isinf)
            range_mask = (
                (depth_numpy >= self.min_depth) & 
                (depth_numpy <= self.max_depth) & 
                (~np.isnan(depth_numpy)) & 
                (~np.isinf(depth_numpy))
            )

            # 5. Combine all masks
            final_mask_numpy = mask_numpy & edge_mask & range_mask

            # 6. Convert back to torch tensor
            mask = torch.from_numpy(final_mask_numpy).bool()
            
            # Update rasters
            rasters["valid_mask_raw"] = mask.clone()
            rasters["valid_mask_filled"] = mask.clone()
            
            # --- END: Modified Mask Logic ---

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other
