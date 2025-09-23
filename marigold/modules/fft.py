import torch
import torch.fft

def enhance_edges_freq(rgb, alpha=1.5, keep_original=True):
    """
    基于频域增强边缘特征
    Args:
        rgb: [B, 3, H, W], 输入归一化 RGB 图像 (-1~1)
        alpha: 高频增强系数 (>1 表示增强边缘)
        keep_original: 是否保留原始 RGB 通道
    
    Returns:
        rgb_out: [B, 3 或 6, H, W]，增强后的图像或拼接
    """
    B, C, H, W = rgb.shape
    device = rgb.device

    # 做 FFT
    fft = torch.fft.fft2(rgb, dim=(2, 3))
    fft_shift = torch.fft.fftshift(fft, dim=(2, 3))  # 把低频移到中心

    # 构建高频增强 mask
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    yy, xx = yy.to(device), xx.to(device)
    center_y, center_x = H // 2, W // 2
    dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)  # 距离中心的半径
    radius = min(H, W) // 4  # 高频阈值：1/4 半径以外算高频
    mask = (dist > radius).float()  # 高频区域=1，低频=0

    # 对高频部分放大
    fft_shift = fft_shift * (1 + (alpha - 1) * mask)

    # 反变换
    fft_unshift = torch.fft.ifftshift(fft_shift, dim=(2, 3))
    img_enhanced = torch.fft.ifft2(fft_unshift, dim=(2, 3)).real  # 取实部

    # 拼接或替换
    if keep_original:
        rgb_out = torch.cat([rgb, img_enhanced], dim=1)  # [B, 6, H, W]
    else:
        rgb_out = img_enhanced  # [B, 3, H, W]

    return rgb_out
