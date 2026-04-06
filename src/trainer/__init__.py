# Author: Bingxin Ke
# Last modified: 2024-05-17

from .apdepth_trainer import ApDepthTrainer


trainer_cls_name_dict = {
    "ApDepthTrainer": ApDepthTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
