import os
import torch
from safetensors.torch import save_file

def convert_bin_to_safetensors(
    bin_path: str,
    output_dir: str,
    output_filename: str = "converted_model.safetensors",
    strict: bool = True
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    state_dict = torch.load(bin_path, map_location="cpu")

    if strict:
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                raise ValueError(f"Error: {k} - {type(v)}")

    save_path = os.path.join(output_dir, output_filename)

    save_file(state_dict, save_path)

    if not os.path.exists(save_path):
        raise RuntimeError(f"File save failed: {save_path}")

    print(f"Conversion completed. File saved to:\n{save_path}")
    return save_path

if __name__ == "__main__":
    bin_file = "/root/Marigold/output/train_marigold/checkpoint/latest/unet/diffusion_pytorch_model.bin"
    output_directory = "/root/Marigold/output/convert"
    output_name = "diffusion_pytorch_model.safetensors"

    try:
        final_path = convert_bin_to_safetensors(
            bin_path=bin_file,
            output_dir=output_directory,
            output_filename=output_name
        )
    except Exception as e:
        print(f"Conversion failed: {str(e)}")