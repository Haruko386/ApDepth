"""Measured far-depth supervision, ported from ApDepth-G to single-step ApDepth."""

import torch
import torch.nn.functional as F


def prepare_far_supervision(depth, valid_mask, far_mask, far_value=1.0, scale=8):
    """Complete targets before VAE encoding; keep metric GT masks unchanged.

    Far masks must come from native dataset depth, never missing GT or DA2.
    Normalization quantiles must already have used only metric-valid GT.
    A latent cell is supported only when every pixel is valid or known far,
    including mixed foreground/far cells. Unknown pixels remain excluded.
    """
    if depth.shape != valid_mask.shape or depth.shape != far_mask.shape:
        raise ValueError("Depth, valid mask and far mask must have identical shapes")
    valid_mask, far_mask = valid_mask.bool(), far_mask.bool()
    far_mask = far_mask & ~valid_mask
    target = torch.where(far_mask, depth.new_tensor(far_value), depth)

    def all_pixels(mask):
        return ~F.max_pool2d((~mask).float(), scale, scale).bool()

    valid = all_pixels(valid_mask)
    support = all_pixels(valid_mask | far_mask)
    added = support & ~valid
    return target, valid, added, support


def masked_loss(loss_fn, prediction, target, mask):
    """Apply a reconstruction criterion with a differentiable zero if empty."""
    selected = prediction.masked_select(mask)
    if selected.numel() == 0:
        return selected.sum()
    return loss_fn(selected, target.masked_select(mask)).mean()
