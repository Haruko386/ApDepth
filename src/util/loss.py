# Last modified: 2025-11-11
#
# Copyright 2025 Jiawei Wang SJZU. All rights reserved.
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
# Please find bibtex at: https://github.com/Haruko386/ApDepth
# More information about the method can be found at https://haruko386.github.io/research
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "latent_freq_loss" == loss_name:
        criterion = LatentFrequencyLoss(**kwargs)
    elif "latent_grad_loss" == loss_name:
        criterion = LatentGradLoss()
    else:
        raise NotImplementedError

    return criterion


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


class LatentFrequencyLoss(nn.Module):
    def __init__(self, loss_type='l1', high_pass_weight=1.0, eps=1e-8):
        super().__init__()
        self.loss_type = loss_type
        self.high_pass_weight = high_pass_weight
        self.eps = eps
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss
        self.weights_cache = {}

    def create_high_pass_weights(self, shape, device):
        if shape in self.weights_cache:
            return self.weights_cache[shape]

        b, c, h, w = shape

        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        center_y, center_x = h // 2, w // 2
        
        dist_from_center = torch.sqrt(
            (y_coords - center_y)**2 + (x_coords - center_x)**2
        )

        max_dist = torch.sqrt(torch.tensor(center_y**2 + center_x**2))
        normalized_dist = dist_from_center / (max_dist + self.eps)
        
        weights = 1.0 + self.high_pass_weight * normalized_dist

        weights = weights.view(1, 1, h, w)
        self.weights_cache[shape] = weights
        return weights

    def forward(self, pred_latent, target_latent, mask=None):
        if mask is not None:
            pred_latent = torch.where(mask, pred_latent, 0.0)
            target_latent = torch.where(mask, target_latent, 0.0)

        fft_pred = torch.fft.fft2(pred_latent, dim=(-2, -1))
        fft_target = torch.fft.fft2(target_latent, dim=(-2, -1))
        
        fft_pred_shifted = torch.fft.fftshift(fft_pred, dim=(-2, -1))
        fft_target_shifted = torch.fft.fftshift(fft_target, dim=(-2, -1))

        fft_pred_mag = torch.abs(fft_pred_shifted)
        fft_target_mag = torch.abs(fft_target_shifted)

        base_loss = self.loss_fn(
            fft_pred_mag, 
            fft_target_mag, 
            reduction='none'
        )

        if self.high_pass_weight > 0:
            weights = self.create_high_pass_weights(
                pred_latent.shape, 
                pred_latent.device
            )
            base_loss = base_loss * weights
            
        if mask is not None:
            return torch.mean(base_loss)
        else:
            return torch.mean(base_loss)
        
class HuberLoss:
    def __init__(self, delta=0.5):
        self.delta = delta
        
    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_gt - depth_pred
        abs_diff = torch.abs(diff)
        squared_diff = diff ** 2
        loss = torch.where(abs_diff > self.delta, 0.5 * squared_diff, self.delta * abs_diff - 0.5 * self.delta ** 2)
        if valid_mask is not None:
            return torch.mean(loss[valid_mask])
        else:
            return torch.mean(loss)
        

class LatentGradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_pred_latent, target_latent, mask=None):
        B, C, H, W = depth_pred_latent.shape
        
        grad_x_pred = torch.abs(depth_pred_latent[..., 1:] - depth_pred_latent[..., :-1])
        grad_x_target = torch.abs(target_latent[..., 1:] - target_latent[..., :-1])

        grad_x_diff = torch.abs(grad_x_pred - grad_x_target)

        grad_y_pred = torch.abs(depth_pred_latent[:, :, 1:, :] - depth_pred_latent[:, :, :-1, :])
        grad_y_target = torch.abs(target_latent[:, :, 1:, :] - target_latent[:, :, :-1, :])

        grad_y_diff = torch.abs(grad_y_pred - grad_y_target)

        latent_grad_loss = 0.0

        if mask is not None:
            mask_x = mask[..., :-1]

            mask_y = mask[:, :, :-1, :]

            latent_grad_loss += (grad_x_diff * mask_x).sum() / mask_x.sum()
            latent_grad_loss += (grad_y_diff * mask_y).sum() / mask_y.sum()
        else:
            latent_grad_loss = grad_x_diff.mean() + grad_y_diff.mean()

        return latent_grad_loss