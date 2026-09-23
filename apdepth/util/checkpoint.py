"""Load Stage-2 trainables on top of the original SD2 pipeline components."""

import json
from pathlib import Path

from diffusers import AutoencoderKL, UNet2DConditionModel


def load_training_components(checkpoint, torch_dtype):
    """Return from_pretrained overrides, including the learned VAE when required.

    Training checkpoints store full U-Net/VAE components, but reuse the base
    SD2 tokenizer, text encoder and scheduler. Never silently discard a VAE.
    """
    root = Path(checkpoint)
    metadata_path = root / "apdepth_training.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    vae_path = root / "vae"
    if metadata.get("has_finetuned_vae", False) and not vae_path.is_dir():
        raise FileNotFoundError(f"Checkpoint requires its fine-tuned VAE: {vae_path}")
    result = {
        "unet": UNet2DConditionModel.from_pretrained(
            root / "unet",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
    }
    if vae_path.is_dir():
        result["vae"] = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
    return result
