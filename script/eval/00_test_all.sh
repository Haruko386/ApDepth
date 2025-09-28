#!/usr/bin/env bash
set -e
set -x

export BASE_DATA_DIR=/root/Dataset

bash /root/Marigold/script/eval/11_infer_nyu.sh
bash /root/Marigold/script/eval/12_eval_nyu.sh

export BASE_DATA_DIR=/root/Datasets

bash /root/Marigold/script/eval/21_infer_kitti.sh
bash /root/Marigold/script/eval/22_eval_kitti.sh

bash /root/Marigold/script/eval/31_infer_eth3d.sh
bash /root/Marigold/script/eval/32_eval_eth3d.sh

bash /root/Marigold/script/eval/41_infer_scannet.sh
bash /root/Marigold/script/eval/42_eval_scannet.sh

bash /root/Marigold/script/eval/51_infer_diode.sh
bash /root/Marigold/script/eval/52_eval_diode.sh

# Since the inference process consumes relatively little GPU memory, 
# a single 4090 with 24GB of VRAM is sufficient to perform parallel inference on five datasets simultaneously across multiple terminals.