# Last modified: 2025-01-14
#
# Copyright 2025 Ziyang Song, USTC. All rights reserved.
#
# This file has been modified from the original version.
# Original copyright (c) 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# Please find bibtex at: https://github.com/indu1ge/DepthMaster#-citation
# More information about the method can be found at https://indu1ge.github.io/DepthMaster_page
# --------------------------------------------------------------------------

import torch
import torch.nn as nn


def get_loss(loss_name, **kwargs):
    # 添加参数安全检查
    safe_kwargs = kwargs if kwargs is not None else {}
    
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**safe_kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**safe_kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**safe_kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**safe_kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**safe_kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "huber_loss" == loss_name:
        delta = safe_kwargs.get('delta', 1.0)
        reduction = safe_kwargs.get('reduction', 'mean')
        criterion = HuberLoss(delta=delta, reduction=reduction)
    elif "mse_grad" == loss_name:
        # 新增：均方 + 梯度损失
        grad_weight = safe_kwargs.get("grad_weight", 1.0)
        reduction = safe_kwargs.get("reduction", "mean")
        criterion = MSEGradLoss(grad_weight=grad_weight, reduction=reduction)
    else:
        raise NotImplementedError

    return criterion

class MSEGradLoss(nn.Module):
    def __init__(self, grad_weight=1.0, reduction="mean"):
        super().__init__()
        self.grad_weight = float(grad_weight)
        assert reduction in ("mean", "sum", None)
        self.reduction = reduction

    @staticmethod
    def _compute_gradients(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    def forward(self, pred, gt, valid_mask=None):
        """
        pred, gt: can be
          - full tensors [B,C,H,W]
          - already masked flattened tensors (1D or N-D where shapes match), in which case valid_mask should be None
        valid_mask: boolean tensor with same shape as pred/gt when they are full [B,C,H,W]
        """
        # case 1: inputs are already flattened/selected (e.g., pred = pred_tensor[mask])
        if valid_mask is None and pred.shape != gt.shape:
            # assume both pred and gt are flattened vectors of same length (由训练代码索引产生)
            diff = pred - gt
            mse = (diff ** 2).mean()
            # no gradient loss can be computed sensibly in this mode => return mse
            total = mse
            if self.reduction == "sum":
                total = mse * diff.numel()
            return total

        # case 2: shapes match (full tensors) -> compute masked or full loss
        if pred.shape != gt.shape:
            raise ValueError("pred and gt must have same shape unless they are already masked vectors.")

        # make sure float
        pred = pred.float()
        gt = gt.float()

        if valid_mask is not None:
            # valid_mask: boolean with same shape [B,C,H,W] or broadcastable
            # mask out invalid elements before computing mse
            # to avoid divide-by-zero, compute number of valid elements
            valid_mask_bool = valid_mask.bool()
            if valid_mask_bool.numel() == 0 or valid_mask_bool.sum() == 0:
                # no valid pixels: return zero loss
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

            diff = pred - gt
            diff_valid = diff[valid_mask_bool]
            mse = (diff_valid ** 2)
            if self.reduction == "mean":
                mse_term = mse.mean()
            elif self.reduction == "sum":
                mse_term = mse.sum()
            else:
                mse_term = mse

            # 梯度损失：在 valid 区域对相邻像素都为 valid 的位置计算
            # 计算 pred/gt 的梯度
            pred_gx, pred_gy = self._compute_gradients(pred)
            gt_gx, gt_gy = self._compute_gradients(gt)

            # 构造对应的 valid mask for gradients（对应相邻对都为 valid）
            # valid_mask shape: [B,C,H,W]
            mask_x = valid_mask_bool[:, :, :, 1:] & valid_mask_bool[:, :, :, :-1]  # 对应 gx
            mask_y = valid_mask_bool[:, :, 1:, :] & valid_mask_bool[:, :, :-1, :]  # 对应 gy

            # 计算 L1 损失在 mask_x, mask_y 上
            gx_diff = torch.abs(pred_gx - gt_gx)
            gy_diff = torch.abs(pred_gy - gt_gy)

            # 如果没有有效的 gradient mask，梯度项为零
            gx_valid = gx_diff[mask_x] if mask_x.sum() > 0 else torch.tensor(0.0, device=pred.device)
            gy_valid = gy_diff[mask_y] if mask_y.sum() > 0 else torch.tensor(0.0, device=pred.device)

            # 聚合
            if isinstance(gx_valid, torch.Tensor) and gx_valid.numel() > 0:
                gx_term = gx_valid.mean() if self.reduction == "mean" else gx_valid.sum()
            else:
                gx_term = torch.tensor(0.0, device=pred.device)

            if isinstance(gy_valid, torch.Tensor) and gy_valid.numel() > 0:
                gy_term = gy_valid.mean() if self.reduction == "mean" else gy_valid.sum()
            else:
                gy_term = torch.tensor(0.0, device=pred.device)

            grad_term = (gx_term + gy_term) * 0.5  # 合并 x,y

            total = mse_term + self.grad_weight * grad_term
            # print("MSE:", mse_term.item(), "Grad:", self.grad_weight * grad_term.item()) # for debug only
            return total

        else:
            # valid_mask is None and pred/gt are full tensors -> compute over all pixels
            diff = pred - gt
            if self.reduction == "mean":
                mse_term = (diff ** 2).mean()
            elif self.reduction == "sum":
                mse_term = (diff ** 2).sum()
            else:
                mse_term = (diff ** 2)

            pred_gx, pred_gy = self._compute_gradients(pred)
            gt_gx, gt_gy = self._compute_gradients(gt)

            gx_diff = torch.abs(pred_gx - gt_gx)
            gy_diff = torch.abs(pred_gy - gt_gy)

            gx_term = gx_diff.mean() if self.reduction == "mean" else gx_diff.sum()
            gy_term = gy_diff.mean() if self.reduction == "mean" else gy_diff.sum()
            grad_term = (gx_term + gy_term) * 0.5

            total = mse_term + self.grad_weight * grad_term
            # print("MSE:", mse_term.item(), "Grad:", self.grad_weight * grad_term.item()) # for debug only
            return total

class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss
    
# class HuberLoss:
#     def __init__(self, delta=0.5):
#         self.delta = delta
        
#     def __call__(self, depth_pred, depth_gt, valid_mask=None):
#         # huber 损失
#         # 计算预测值与真实值的差值
#         diff = depth_gt - depth_pred
        
#         # 计算绝对值和差值的平方
#         abs_diff = torch.abs(diff)
#         squared_diff = diff ** 2
        
#         # 使用条件语句选择L2损失或L1损失
#         loss = torch.where(abs_diff > self.delta, 0.5 * squared_diff, self.delta * abs_diff - 0.5 * self.delta ** 2)
        
#         # 返回所有样本损失的总和
#         if valid_mask is not None:
#             return torch.mean(loss[valid_mask])
#         else:
#             return torch.mean(loss)

class HuberLoss(nn.Module):
    def __init__(self, delta=0.5, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, depth_pred, depth_gt, valid_mask=None):
        # 计算预测值与真实值的差值
        diff = depth_gt - depth_pred
        
        # 计算绝对差值
        abs_diff = torch.abs(diff)
        
        # 当 |diff| <= delta 时，使用平方项
        # 当 |diff| > delta 时，使用线性项
        loss = torch.where(
            abs_diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        
        # 应用有效掩码（如果提供）
        if valid_mask is not None:
            loss = loss[valid_mask]
        
        return loss.mean()

class SILogRMSELoss:
    def __init__(self, lamb, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        # self.alpha = alpha
        self.pred_in_log = log_pred

    # def __call__(self, depth_pred, depth_gt, valid_mask):
    #     log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
    #     log_depth_gt = torch.log(depth_gt)
    #     # borrowed from https://github.com/aliyun/NeWCRFs
    #     # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
    #     # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

    #     diff = log_depth_pred - log_depth_gt
    #     if valid_mask is not None:
    #         diff[~valid_mask] = 0
    #         n = valid_mask.sum((-1, -2))
    #     else:
    #         n = depth_gt.shape[-2] * depth_gt.shape[-1]

    #     diff2 = torch.pow(diff, 2)
    #     first_term = torch.sum(diff2, (-1, -2)) / n
    #     second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    #     loss = torch.sqrt(first_term - second_term).mean()
    #     return loss
    def __call__(self, depth_pred, depth_gt, valid_mask):
        valid_mask = valid_mask.detach()
        log_depth_pred = torch.log(depth_pred[valid_mask])
        log_depth_gt = torch.log(depth_gt[valid_mask])

        diff = log_depth_gt - log_depth_pred

        first_term = torch.pow(diff, 2).mean()
        second_term = self.lamb * torch.pow(diff.mean(), 2)
        loss = torch.sqrt(first_term - second_term)
        return loss


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_disp = disp.mean(2, True).mean(3, True)
    disp = disp / (mean_disp + 1e-7)
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)