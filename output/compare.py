import numpy as np
from safetensors import safe_open
import torch

def compare_bin_safetensor(bin_file_path, safetensor_file_path):
    # 读取.bin文件 (PyTorch格式)
    bin_model = torch.load(bin_file_path, map_location='cpu')
    
    # 读取.safetensors文件
    safetensor_model = {}
    with safe_open(safetensor_file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            safetensor_model[key] = f.get_tensor(key)
    
    # 比较键的数量
    bin_keys = set(bin_model.keys())
    safetensor_keys = set(safetensor_model.keys())
    
    if bin_keys != safetensor_keys:
        print("键不匹配!")
        print(f".bin文件有但.safetensor文件没有的键: {bin_keys - safetensor_keys}")
        print(f".safetensor文件有但.bin文件没有的键: {safetensor_keys - bin_keys}")
        return False
    
    # 比较每个参数
    all_match = True
    for key in bin_keys:
        bin_param = bin_model[key].numpy()
        safetensor_param = safetensor_model[key].numpy()
        
        if bin_param.shape != safetensor_param.shape:
            print(f"形状不匹配: {key}")
            print(f".bin文件形状: {bin_param.shape}")
            print(f".safetensor文件形状: {safetensor_param.shape}")
            all_match = False
            continue
        
        if not np.allclose(bin_param, safetensor_param, atol=1e-6):
            print(f"值不匹配: {key}")
            max_diff = np.max(np.abs(bin_param - safetensor_param))
            print(f"最大差异: {max_diff}")
            all_match = False
    
    if all_match:
        print("所有参数完全匹配!")
        return True
    else:
        print("参数存在差异!")
        return False

# 使用示例
if __name__ == "__main__":
    bin_file = "/root/Marigold/output/train_marigold/checkpoint/latest/unet/diffusion_pytorch_model.bin"  # 替换为你的.bin文件路径
    # bin_file = "/root/Marigold_output/备份（第一次修改FFT）/train_marigold/checkpoint/latest/unet/diffusion_pytorch_model.bin"  # 替换为你的.bin文件路径
    safetensor_file = "/root/.cache/huggingface/hub/models--prs-eth--marigold-v1-0/snapshots/f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c/unet/diffusion_pytorch_model.safetensors"  # 替换为你的.safetensors文件路径
    # safetensor_file = "/root/Marigold/output/convert/diffusion_pytorch_model.safetensors"
    print(f"比较 {bin_file} 和 {safetensor_file}:")
    result = compare_bin_safetensor(bin_file, safetensor_file)
    
    if result:
        print("文件内容完全相同")
    else:
        print("文件内容存在差异")