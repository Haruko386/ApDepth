import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeExtractor(nn.Module):
    """
    可微分的 Sobel 边缘提取模块
    输入:  RGB [N,3,H,W]
    输出:  边缘强度 [N,1,H,W] (范围0~1)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        # Sobel 核，固定
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1,  2,  1],
                           [0,  0,  0],
                           [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))
        self.eps = eps

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        device = rgb.device  # 获取输入所在设备
        kx = self.kx.to(device)
        ky = self.ky.to(device)

        # 灰度化
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [N,1,H,W]

        # Sobel梯度
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)  # [N,1,H,W]

        # 每图 min-max 归一化 (用分位数截断避免极端值)
        B = mag.size(0)
        mag_flat = mag.view(B, -1)
        lo = torch.quantile(mag_flat, 0.02, dim=1, keepdim=True).view(B,1,1,1)
        hi = torch.quantile(mag_flat, 0.98, dim=1, keepdim=True).view(B,1,1,1)
        norm = (mag - lo) / (hi - lo + self.eps)
        return norm.clamp(0, 1)



class LatentFuser(nn.Module):
    """
    融合器: 输入12通道 [rgb(4)+edge(4)+noisy(4)] -> 输出8通道
    初始参数设计为透传rgb和noisy，edge分支初始权重为0
    """
    def __init__(self):
        super().__init__()
        self.latent_fuser = nn.Conv2d(12, 8, kernel_size=1, bias=True)

        # 初始化权重
        with torch.no_grad():
            self.latent_fuser.weight.zero_()
            self.latent_fuser.bias.zero_()
            # rgb映射到前4通道
            for c in range(4):
                self.latent_fuser.weight[c, c, 0, 0] = 1.0
            # noisy映射到后4通道
            for c in range(4):
                self.latent_fuser.weight[4 + c, 8 + c, 0, 0] = 1.0

    def forward(self, x):
        """
        x: [N,12,H,W]
        顺序: [rgb(0..3), edge(4..7), noisy(8..11)]
        """
        return self.latent_fuser(x)
