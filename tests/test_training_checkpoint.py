"""Tests for composing saved training components with the original SD2 base."""

import json

import pytest
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel

from apdepth.util.checkpoint import load_training_components


def tiny_unet():
    return UNet2DConditionModel(
        sample_size=4,
        in_channels=8,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(16,),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        cross_attention_dim=16,
        norm_num_groups=8,
    )


def tiny_vae():
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(16, 16, 16, 16),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        norm_num_groups=8,
    )


def test_loads_saved_unet_and_vae(tmp_path):
    unet = tiny_unet()
    vae = tiny_vae()
    unet.save_pretrained(tmp_path / "unet", safe_serialization=True)
    vae.save_pretrained(tmp_path / "vae", safe_serialization=True)
    (tmp_path / "apdepth_training.json").write_text(
        json.dumps({"has_finetuned_vae": True}), encoding="utf-8"
    )

    components = load_training_components(tmp_path, torch.float32)
    assert set(components) == {"unet", "vae"}
    torch.testing.assert_close(
        components["unet"].conv_in.weight, unet.conv_in.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        components["vae"].decoder.conv_out.weight,
        vae.decoder.conv_out.weight,
        rtol=0,
        atol=0,
    )


def test_loads_legacy_unet_only_checkpoint(tmp_path):
    tiny_unet().save_pretrained(tmp_path / "unet", safe_serialization=True)
    assert set(load_training_components(tmp_path, torch.float32)) == {"unet"}


def test_rejects_missing_required_vae(tmp_path):
    tiny_unet().save_pretrained(tmp_path / "unet", safe_serialization=True)
    (tmp_path / "apdepth_training.json").write_text(
        json.dumps({"has_finetuned_vae": True}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="fine-tuned VAE"):
        load_training_components(tmp_path, torch.float32)
