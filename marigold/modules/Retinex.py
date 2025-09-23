import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.utils import save_image

# 修改 RetinexEnhancementNet 类
class RetinexEnhancementNet(nn.Module):
    def __init__(self, save_debug_path="/root/2/img"):
        super(RetinexEnhancementNet, self).__init__()
        self.save_debug_path = save_debug_path
        
        # 创建保存目录
        os.makedirs(save_debug_path, exist_ok=True)
        
        # 估计光照分量
        self.illumination_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 反射分量增强
        self.reflectance_enhance = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
        
        # 计数器用于保存不同的图像
        self.save_counter = 0
        
    def forward(self, x):
        # 保存原始图像（在增强前）
        # self.save_original_images(x)
        # y = x.clone()
        
        # 估计光照分量
        illumination = self.illumination_net(x)
        
        # 计算反射分量 (R = I / L)
        reflectance = x / (illumination + 1e-8)
        
        # 增强反射分量
        enhanced_reflectance = self.reflectance_enhance(reflectance)
        
        # 重新组合
        enhanced = illumination * enhanced_reflectance
        enhanced = torch.clamp(enhanced, 0, 1)

        enhanced = enhanced
        # 保存增强后的图像
        # self.save_enhanced_images(enhanced)
        
        return enhanced
    
    def save_original_images(self, original):
        """保存原始图像"""
        if self.save_counter < 20:  # 最多保存20张
            for i in range(min(4, original.shape[0])):  # 每个batch保存最多4张图像
                orig_img = original[i].cpu()
                # 反归一化（如果图像是归一化的）
                if orig_img.min() < 0 or orig_img.max() > 1:
                    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
                
                save_path = os.path.join(self.save_debug_path, f"orig_{self.save_counter:03d}_i{i}.png")
                save_image(orig_img, save_path)
    
    def save_enhanced_images(self, enhanced):
        """保存增强后的图像"""
        if self.save_counter < 20:
            for i in range(min(4, enhanced.shape[0])):
                enh_img = enhanced[i].cpu()
                save_path = os.path.join(self.save_debug_path, f"enhanced_{self.save_counter:03d}_i{i}.png")
                save_image(enh_img, save_path)
            
            self.save_counter += 1