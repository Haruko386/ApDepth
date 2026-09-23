"""CPU tests for SDWT and the direct-SD2 training path."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel

from src.trainer.apdepth_trainer import ApDepthTrainer
from src.util.config_util import recursive_load_config
from src.util.sdwt_loss import (
    SD2SDWTObjective,
    SpatiallyDebiasedWindowTransportLoss,
    masked_gradient_loss,
)


torch.set_num_threads(1)


def test_debiased_transport_is_zero_on_identical_depth():
    torch.manual_seed(4)
    pred = torch.randn(2, 1, 7, 11, requires_grad=True)
    mask = torch.rand_like(pred) > 0.3
    loss = SpatiallyDebiasedWindowTransportLoss(chunk_size=2)(pred, pred, mask)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2e-6, rtol=0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_spatial_cost_distinguishes_near_and_far_permutations():
    target = torch.zeros(1, 1, 5, 5)
    target[..., 2, 2] = 1.0
    near = torch.zeros_like(target)
    near[..., 2, 3] = 1.0
    far = torch.zeros_like(target)
    far[..., 0, 0] = 1.0
    criterion = SpatiallyDebiasedWindowTransportLoss(shifted_windows=False)
    mask = torch.ones_like(target, dtype=torch.bool)
    assert criterion(near, target, mask) < criterion(far, target, mask)


def test_shifted_grid_reduces_fixed_window_seam_penalty():
    target = torch.zeros(1, 1, 5, 10)
    target[..., 2, 4] = 1.0
    pred = torch.zeros_like(target)
    pred[..., 2, 5] = 1.0
    mask = torch.ones_like(target, dtype=torch.bool)
    fixed = SpatiallyDebiasedWindowTransportLoss(shifted_windows=False)(
        pred, target, mask
    )
    dual = SpatiallyDebiasedWindowTransportLoss(shifted_windows=True)(
        pred, target, mask
    )
    assert dual < fixed


def test_empty_and_nan_invalid_targets():
    pred = torch.randn(1, 1, 3, 7, requires_grad=True)
    target = torch.full_like(pred, float("nan"))
    mask = torch.zeros_like(pred, dtype=torch.bool)
    criterion = SpatiallyDebiasedWindowTransportLoss()
    loss = criterion(pred, target, mask)
    loss.backward()
    assert loss == 0
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_chunking_does_not_change_loss_or_gradient():
    torch.manual_seed(1)
    pred = torch.randn(2, 1, 11, 13, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.rand_like(pred) > 0.3
    small = SpatiallyDebiasedWindowTransportLoss(chunk_size=1)(pred, target, mask)
    large = SpatiallyDebiasedWindowTransportLoss(chunk_size=1024)(pred, target, mask)
    torch.testing.assert_close(small, large)
    torch.testing.assert_close(
        torch.autograd.grad(small, pred, retain_graph=True)[0],
        torch.autograd.grad(large, pred)[0],
    )


def test_gradient_loss_ignores_invalid_pairs():
    pred = torch.tensor([[[[1.0, 3.0, 900.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 0.0, float("nan")]]]])
    mask = torch.tensor([[[[True, True, False]]]])
    loss = masked_gradient_loss(pred, target, mask)
    assert loss == 2
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[[-1.0, 1.0, 0.0]]]]))


@pytest.mark.parametrize(
    "step,weight",
    [(0, 0.0), (8000, 0.0), (10000, 0.5), (12000, 1.0), (21000, 1.0)],
)
def test_curriculum_keeps_reconstruction_supervision(step, weight):
    pred = torch.randn(1, 1, 5, 5, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss, terms = SD2SDWTObjective()(pred, target, pred, target, mask, mask, step)
    assert terms["sdwt_weight"] == weight
    assert terms["latent_weight"] >= 0.5
    assert terms["pixel_weight"] >= 0.25
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_all_sdwt_configs_resolve():
    configs = [
        "config/train_sd2_sdwt.yaml",
        "config/train_sd2_sdwt_fft.yaml",
        "config/ablation/sd2_sdwt_equal_windows.yaml",
        "config/ablation/sd2_sdwt_no_debias.yaml",
        "config/ablation/sd2_sdwt_no_decoder.yaml",
        "config/ablation/sd2_sdwt_no_spatial.yaml",
        "config/ablation/sd2_sdwt_no_transport.yaml",
        "config/ablation/sd2_sdwt_single_grid.yaml",
    ]
    for path in configs:
        cfg = recursive_load_config(path)
        assert cfg.initialization == "base_sd2"
        assert cfg.sdwt_loss.enabled


class TinyPipeline:
    def __init__(self):
        self.unet = UNet2DConditionModel(
            sample_size=4,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(16,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            cross_attention_dim=16,
            norm_num_groups=8,
        )
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(16, 16, 16, 16),
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            norm_num_groups=8,
        )
        self.text_encoder = torch.nn.Linear(16, 16)
        self.da2 = SimpleNamespace(infer_batch=lambda rgb: rgb.detach())

    def encode_empty_text(self):
        self.empty_text_embed = torch.zeros(1, 2, 16)

    def decode_depth(self, latent):
        return self.vae.decode(latent / 0.18215).sample.mean(1, keepdim=True)


def make_trainer(tmp_path):
    cfg = recursive_load_config("config/train_sd2_sdwt.yaml")
    cfg.trainer.enable_xformers = False
    cfg.trainer.gradient_checkpointing = False
    cfg.lr_scheduler.kwargs.warmup_steps = 0
    cfg.vae_decoder.start_iter = 2
    cfg.sdwt_loss.kwargs.start_iter = 1
    cfg.sdwt_loss.kwargs.ramp_iters = 1
    return ApDepthTrainer(
        cfg,
        TinyPipeline(),
        [],
        torch.device("cpu"),
        "",
        str(tmp_path),
        str(tmp_path),
        str(tmp_path),
        1,
        [],
        [],
    )


def test_delayed_decoder_and_checkpoint_round_trip(tmp_path):
    trainer = make_trainer(tmp_path)
    assert trainer.model.unet.config.in_channels == 8
    trainer._set_training_mode()
    assert not any(p.requires_grad for p in trainer.model.vae.parameters())

    trainer.effective_iter = 2
    trainer._set_training_mode()
    assert all(p.requires_grad for p in trainer.decoder_params)
    assert not any(p.requires_grad for p in trainer.model.vae.encoder.parameters())

    x = torch.randn(1, 8, 4, 4)
    latent = trainer.model.unet(x, 1, trainer.empty_text_embed).sample
    pred = trainer.model.decode_depth(latent)
    loss, *_ = trainer._compute_depth_loss(
        latent,
        torch.zeros_like(latent),
        pred,
        torch.zeros_like(pred),
        torch.ones_like(latent, dtype=torch.bool),
        torch.ones_like(pred, dtype=torch.bool),
        torch.zeros_like(latent, dtype=torch.bool),
        torch.zeros_like(pred, dtype=torch.bool),
    )
    loss.backward()
    assert trainer.model.unet.conv_out.weight.grad.abs().sum() > 0
    assert trainer.model.vae.decoder.conv_out.weight.grad.abs().sum() > 0

    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    trainer.save_checkpoint("latest", save_train_state=True)

    clone = make_trainer(tmp_path / "clone")
    clone.load_checkpoint(str(tmp_path / "latest"))
    assert clone.effective_iter == trainer.effective_iter
    assert clone.has_finetuned_vae
    for current, restored in zip(
        trainer.model.vae.parameters(), clone.model.vae.parameters()
    ):
        torch.testing.assert_close(current, restored, rtol=0, atol=0)
