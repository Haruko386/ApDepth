# Copyright 2026 Jiawei Wang, SJZU.
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).
"""Spatially Debiased Window Transport for direct SD2 depth training.

SDWT keeps the useful tolerance of local optimal-transport matching while
constraining displacement, removing entropic self-bias, reducing tiling seams,
and weighting windows by their valid support.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .far_supervision import masked_loss


class SpatiallyDebiasedWindowTransportLoss(nn.Module):
    """Compare local depth distributions without discarding pixel geometry.

    A transport plan is fitted with a joint depth-and-position cost. The final
    value is an entropic Sinkhorn divergence, so identical inputs have zero
    loss despite the entropy regularizer. A second, half-window-shifted tiling
    lets structures cross the boundaries of the first tiling.
    """

    def __init__(
        self,
        window_size=5,
        transport_iter=10,
        transport_tau=0.1,
        spatial_weight=0.15,
        charbonnier_eps=1.0e-3,
        debiased=True,
        shifted_windows=True,
        support_weighted=True,
        chunk_size=512,
    ):
        super().__init__()
        for name, value, minimum in (
            ("window_size", window_size, 2),
            ("transport_iter", transport_iter, 1),
            ("chunk_size", chunk_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value, positive in (
            ("transport_tau", transport_tau, True),
            ("spatial_weight", spatial_weight, False),
            ("charbonnier_eps", charbonnier_eps, True),
        ):
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                relation = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {relation}")
        for name, value in (
            ("debiased", debiased),
            ("shifted_windows", shifted_windows),
            ("support_weighted", support_weighted),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")

        self.window_size = window_size
        self.transport_iter = transport_iter
        self.transport_tau = transport_tau
        self.spatial_weight = spatial_weight
        self.charbonnier_eps = charbonnier_eps
        self.debiased = debiased
        self.shifted_windows = shifted_windows
        self.support_weighted = support_weighted
        self.chunk_size = chunk_size

    def _pair_cost(self, source, target, source_xy, target_xy):
        delta = source.unsqueeze(-1) - target.unsqueeze(-2)
        depth_cost = torch.sqrt(delta.square() + self.charbonnier_eps**2)
        depth_cost = depth_cost - self.charbonnier_eps
        spatial_cost = torch.cdist(source_xy, target_xy, p=1)
        return depth_cost + self.spatial_weight * spatial_cost

    def _regularized_transport(self, source, target, source_xy, target_xy):
        cost = self._pair_cost(source, target, source_xy, target_xy)
        log_plan = -cost / self.transport_tau
        log_marginal = -math.log(source.shape[-1])
        for _ in range(self.transport_iter):
            log_plan = log_plan - log_plan.logsumexp(-1, keepdim=True) + log_marginal
            log_plan = log_plan - log_plan.logsumexp(-2, keepdim=True) + log_marginal
        plan = log_plan.exp()
        # Entropic OT objective. The constant -tau term cancels in divergence.
        transport_cost = (plan * cost).sum((-1, -2))
        regularized_objective = transport_cost + self.transport_tau * (
            plan * log_plan
        ).sum((-1, -2))
        return regularized_objective, transport_cost

    def _group_loss(self, pred, target, coordinates):
        cross, cross_cost = self._regularized_transport(
            pred, target, coordinates, coordinates
        )
        if not self.debiased:
            return cross_cost
        pred_self, _ = self._regularized_transport(pred, pred, coordinates, coordinates)
        # The target self-term has no trainable path; avoiding its graph saves
        # memory while preserving the exact divergence value.
        with torch.no_grad():
            target_self, _ = self._regularized_transport(
                target, target, coordinates, coordinates
            )
        # Finite Sinkhorn iterations can leave tiny negative round-off values.
        return (cross - 0.5 * (pred_self + target_self)).clamp_min(0.0)

    def _blocks(self, tensor, offset):
        k = self.window_size
        b, c, h, w = tensor.shape
        pad_right = (-(w + offset)) % k
        pad_bottom = (-(h + offset)) % k
        tensor = F.pad(tensor, (offset, pad_right, offset, pad_bottom))
        return (
            tensor.reshape(
                b,
                c,
                tensor.shape[-2] // k,
                k,
                tensor.shape[-1] // k,
                k,
            )
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, k * k, c)
        )

    def _tiling_loss(self, pred, target, mask, offset):
        pred_blocks = self._blocks(pred, offset)[..., 0]
        target_blocks = self._blocks(target, offset)[..., 0]
        mask_blocks = self._blocks(mask, offset)[..., 0].bool()
        counts = mask_blocks.sum(-1)
        nonempty = counts > 0
        pred_blocks = pred_blocks[nonempty]
        target_blocks = target_blocks[nonempty]
        mask_blocks = mask_blocks[nonempty]
        counts = counts[nonempty]

        total = pred.sum() * 0.0
        support = counts.sum().to(dtype=pred.dtype)
        if counts.numel() == 0:
            return total, support

        k = self.window_size
        axis = torch.arange(k, device=pred.device, dtype=pred.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        normalizer = max(2 * (k - 1), 1)
        base_xy = torch.stack((yy, xx), dim=-1).reshape(k * k, 2) / normalizer
        order = torch.argsort(mask_blocks.to(torch.int8), dim=-1, descending=True)

        # Compacting by support size excludes invalid rows and columns exactly,
        # instead of relying on a large sentinel cost.
        for valid_count in range(1, k * k + 1):
            selected = counts == valid_count
            if not selected.any():
                continue
            indices = order[selected, :valid_count]
            p = pred_blocks[selected].gather(1, indices)
            t = target_blocks[selected].gather(1, indices)
            xy = base_xy.expand(len(indices), -1, -1).gather(
                1, indices.unsqueeze(-1).expand(-1, -1, 2)
            )
            for start in range(0, len(p), self.chunk_size):
                args = (
                    p[start : start + self.chunk_size],
                    t[start : start + self.chunk_size],
                    xy[start : start + self.chunk_size],
                )
                if torch.is_grad_enabled() and p.requires_grad:
                    values = checkpoint(self._group_loss, *args, use_reentrant=False)
                else:
                    values = self._group_loss(*args)
                block_weight = valid_count if self.support_weighted else 1
                total = total + values.sum() * block_weight
        if not self.support_weighted:
            support = counts.new_tensor(len(counts), dtype=pred.dtype)
        return total, support

    def forward(self, pred, target, mask):
        if pred.ndim != 4 or pred.shape[1] != 1:
            raise ValueError("SDWT Loss expects depth [B, 1, H, W]")
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError("Prediction, target and mask shapes must match")

        mask = mask.bool()
        # Invalid NaN/Inf targets must not leak through multiplication by zero.
        pred = torch.where(mask, pred.float(), 0.0)
        target = torch.where(mask, target.float(), 0.0)
        offsets = (0, self.window_size // 2) if self.shifted_windows else (0,)
        total = pred.sum() * 0.0
        support = total.detach().clone()
        for offset in offsets:
            value, count = self._tiling_loss(pred, target, mask, offset)
            total = total + value
            support = support + count
        return total / support.clamp_min(1)


def masked_gradient_loss(pred, target, mask):
    """First differences with both endpoints valid, averaged over valid pairs."""
    mask = mask.bool()
    residual = torch.where(mask, pred.float(), 0.0) - torch.where(
        mask, target.float(), 0.0
    )
    dy = residual[..., 1:, :] - residual[..., :-1, :]
    dx = residual[..., :, 1:] - residual[..., :, :-1]
    my = mask[..., 1:, :] & mask[..., :-1, :]
    mx = mask[..., :, 1:] & mask[..., :, :-1]
    return (dy.masked_select(my).abs().sum() + dx.masked_select(mx).abs().sum()) / (
        my.sum() + mx.sum()
    ).clamp_min(1)


class SD2SDWTObjective(nn.Module):
    """Learn base geometry first, then introduce SDWT in one training run."""

    def __init__(
        self,
        start_iter=8000,
        ramp_iters=4000,
        weight=1.0,
        latent_weight=1.0,
        pixel_weight=1.0,
        gradient_weight=0.5,
        final_latent_weight=0.5,
        final_pixel_weight=0.25,
        final_gradient_weight=0.1,
        **transport_kwargs,
    ):
        super().__init__()
        if start_iter < 0 or ramp_iters <= 0:
            raise ValueError("SDWT needs start_iter >= 0 and ramp_iters > 0")
        self.start_iter, self.ramp_iters = start_iter, ramp_iters
        self.initial = (latent_weight, pixel_weight, gradient_weight)
        self.final = (final_latent_weight, final_pixel_weight, final_gradient_weight)
        if any(
            not math.isfinite(v) or v < 0 for v in (*self.initial, *self.final, weight)
        ):
            raise ValueError("Loss weights must be finite and nonnegative")
        if sum(self.initial[:2]) <= 0 or sum(self.final[:2]) <= 0:
            raise ValueError(
                "SD2 training requires reconstruction supervision throughout"
            )
        self.weight = weight
        self.transport = SpatiallyDebiasedWindowTransportLoss(**transport_kwargs)

    def forward(
        self, pred_latent, target_latent, pred, target, valid_latent, valid_pixel, step
    ):
        progress = min(max((step - self.start_iter) / self.ramp_iters, 0.0), 1.0)
        weights = [a + (b - a) * progress for a, b in zip(self.initial, self.final)]
        latent = masked_loss(
            F.mse_loss, pred_latent.float(), target_latent.float(), valid_latent
        )
        pixel = masked_loss(F.l1_loss, pred.float(), target.float(), valid_pixel)
        gradient = masked_gradient_loss(pred, target, valid_pixel)
        transport_weight = self.weight * progress
        transport = (
            self.transport(pred, target, valid_pixel)
            if transport_weight > 0
            else pred.new_zeros(())
        )
        loss = (
            weights[0] * latent
            + weights[1] * pixel
            + weights[2] * gradient
            + transport_weight * transport
        )
        return loss, {
            "latent_mse": latent.detach().item(),
            "pixel_l1": pixel.detach().item(),
            "gradient_loss": gradient.detach().item(),
            "sdwt_loss": transport.detach().item(),
            "sdwt_weight": transport_weight,
            "latent_weight": weights[0],
            "pixel_weight": weights[1],
            "gradient_weight": weights[2],
        }
