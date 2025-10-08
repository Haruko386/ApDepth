import os
import torch
from safetensors.torch import save_file

def convert_bin_to_safetensors(
    bin_path: str,
    output_dir: str,
    output_filename: str = "converted_model.safetensors",
    strict: bool = True
) -> str:
    """
    将 .bin 文件转换为 .safetensors 格式并保存到指定路径

    Args:
        bin_path: 输入的 .bin 文件路径
        output_dir: 输出目录路径
        output_filename: 输出文件名（需包含.safetensors后缀）
        strict: 是否严格检查权重完整性

    Returns:
        最终保存的完整路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始权重
    state_dict = torch.load(bin_path, map_location="cpu")

    # 验证权重完整性
    if strict:
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                raise ValueError(f"非张量权重: {k} - {type(v)}")

    # 构建输出路径
    save_path = os.path.join(output_dir, output_filename)

    # 转换保存为 .safetensors
    save_file(state_dict, save_path)

    # 验证文件是否生成
    if not os.path.exists(save_path):
        raise RuntimeError(f"文件保存失败: {save_path}")

    print(f"✅ 转换完成！文件已保存到:\n{save_path}")
    return save_path

# 使用示例
if __name__ == "__main__":
    # 示例路径 - 替换为你的实际路径
    bin_file = "/root/Marigold/output/train_marigold/checkpoint/latest/unet/diffusion_pytorch_model.bin"
    # bin_file = "/root/Marigold_output/10. 单步推理第二次尝试(6000轮次换了损失函数)/train_marigold/checkpoint/latest/unet/diffusion_pytorch_model.bin"
    output_directory = "/root/Marigold/output/convert"
    output_name = "diffusion_pytorch_model.safetensors"

    # 执行转换
    try:
        final_path = convert_bin_to_safetensors(
            bin_path=bin_file,
            output_dir=output_directory,
            output_filename=output_name
        )
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")