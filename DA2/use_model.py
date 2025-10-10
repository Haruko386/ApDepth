import cv2
import torch
import numpy as np
import os
from depth_anything_v2.dpt import DepthAnythingV2

# ---------------------------
# 设备选择
# ---------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else \
         'mps' if torch.backends.mps.is_available() else 'cpu'

# ---------------------------
# 模型加载
# ---------------------------
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

encoder = 'vits'  # 可选：'vits'、'vitb'、'vitl'、'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# ---------------------------
# 输入图像路径与输出路径
# ---------------------------
input_path = '/root/1/img/orig_001_i0.png'  # 输入图像路径
output_dir = '/root/1/depth_output'  # 输出目录
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 读取图像并推理深度
# ---------------------------
raw_img = cv2.imread(input_path)
if raw_img is None:
    raise FileNotFoundError(f"未找到图像文件: {input_path}")

depth = model.infer_image(raw_img)  # H×W numpy array，float类型原始深度图

# ---------------------------
# 保存原始深度为 .npy 文件
# ---------------------------
npy_path = os.path.join(output_dir, 'depth.npy')
np.save(npy_path, depth)

# ---------------------------
# 将深度图转换为可视化图片并保存
# ---------------------------
depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # 归一化
depth_vis = (depth_vis * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
cv2.imwrite(os.path.join(output_dir, 'depth_vis.png'), depth_color)

# ---------------------------
# 转换为 [B, 3, H, W] Tensor
# ---------------------------
# 这里我们把单通道深度图扩展为3通道
depth_tensor = torch.tensor(depth_vis, dtype=torch.float32) / 255.0  # [H, W]
depth_tensor = depth_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1, 3, H, W]

# ---------------------------
# 可选择保存为Tensor文件（.pt）
# ---------------------------
torch.save(depth_tensor, os.path.join(output_dir, 'depth_tensor.pt'))

print(f"✅ 深度推理完成：")
print(f" - 原始深度: {npy_path}")
print(f" - 可视化图像: {os.path.join(output_dir, 'depth_vis.png')}")
print(f" - Tensor文件: {os.path.join(output_dir, 'depth_tensor.pt')}")
print(f"Tensor形状: {depth_tensor.shape}")
