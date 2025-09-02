import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTConfig


class EmbeddingAdapter(nn.Module):
    """嵌入适配器模块"""
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )
    
    def forward(self, x, gamma):
        return self.fc(x) * gamma


class CIDE(nn.Module):
    def __init__(self, emb_dim=512, no_of_classes=1000, train_from_scratch=False, 
                 vit_model_name="google/vit-base-patch16-224"):
        super().__init__()
        
        # ViT模型
        if train_from_scratch:
            vit_config = ViTConfig(num_labels=1000)
            self.vit_model = ViTForImageClassification(vit_config)
        else:
            self.vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
        
        # 冻结ViT参数
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, no_of_classes)
        )
        
        self.dim = emb_dim
        self.softmax = nn.Softmax(dim=1)
        
        # 类别嵌入
        self.embeddings = nn.Parameter(torch.randn(no_of_classes, emb_dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=emb_dim)
        
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(emb_dim) * 1e-4)
        
        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, 4 * 16 * 16),  # 默认输出16x16特征
            nn.Unflatten(1, (4, 16, 16))
        )

    def forward(self, x, target_size=None):
        """
        输入: [B, 3, H, W]
        输出: [B, 4, target_H, target_W] (和rgb_latent一致)
        """
        # 反归一化
        x_denorm = (x + 1) / 2
        x_denorm = x_denorm.clamp(0, 1)
        
        # 调整到ViT输入尺寸
        x_resized = F.interpolate(x_denorm, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 标准化
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(x.device)
        x_preprocessed = (x_resized - mean) / std
        
        # ViT提取特征
        with torch.no_grad():
            vit_outputs = self.vit_model(x_preprocessed)
            vit_logits = vit_outputs.logits
        
        # 分类概率
        class_probs = self.fc(vit_logits)
        class_probs = self.softmax(class_probs)
        
        # 类别嵌入
        class_embeddings = class_probs @ self.embeddings
        
        # 适配器
        conditioning_embedding = self.embedding_adapter(class_embeddings, self.gamma)
        
        # 投影
        latent_features = self.feature_projection(conditioning_embedding)  # [B, 4, 16, 16]
        
        # 动态调整到 target_size
        if target_size is not None:
            latent_features = F.interpolate(latent_features, size=target_size, mode='bilinear', align_corners=False)
        
        return latent_features


class CIDELatentFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1)
        )
        with torch.no_grad():
            # RGB通道的初始连接
            for c in range(4):
                self.fusion_conv[0].weight[c, c, 1, 1] = 1.0
            # CIDE特征通道的初始连接
            for c in range(4):
                self.fusion_conv[0].weight[4 + c, 4 + c, 1, 1] = 0.5
            # Noisy通道的初始连接
            for c in range(4):
                self.fusion_conv[0].weight[8 + c, 8 + c, 1, 1] = 1.0

    def forward(self, x):
        return self.fusion_conv(x)