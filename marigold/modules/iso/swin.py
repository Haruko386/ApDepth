import timm
import torch.nn as nn
import torch.nn.functional as F
import torch

class SwinFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("swinv2_base_window12to24_192to384.ms_in22k_ft_in1k", pretrained=True, features_only=True, out_indices=(0,1,2,3), img_size=480)
        # self.backbone.default_cfg['input_size'] = (3, 480, 640)  # 修改默认输入尺寸
        # self.backbone.patch_embed.img_size = (480, 640)          # 覆盖内部约束

        # 输出维度，例如 [96, 192, 384, 768]
        self.out_channels = self.backbone.feature_info.channels()

        # 用 1x1 conv 把每层通道调整到统一维度，比如 4
        self.projs = nn.ModuleList([
            nn.Conv2d(c, 4, kernel_size=1) for c in self.out_channels
        ])

    def forward(self, x):
        feats = self.backbone(x)  # list: [B, C_i, H_i, W_i]
        feats = [proj(f) for f, proj in zip(feats, self.projs)]
        return feats  # list of 4 tensors

class MultiScaleAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        # feats: list of 4 tensors [B, 4, H_i, W_i]
        target_size = feats[0].shape[2:]  # H, W
        feats_up = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                    for f in feats]
        fused = torch.stack(feats_up, dim=1)  # [B, 4(levels), 4(channels), H, W]
        fused = fused.mean(dim=1)  # 简单平均，也可以concat+conv
        return fused  # [B, 4, H, W]