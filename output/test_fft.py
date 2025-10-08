import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def rgb_fft_image(input_path: str, output_path: str, highpass_radius: int = 30):
    """
    对指定路径的 RGB 图像做 FFT 高通滤波处理，并保存结果
    :param input_path: 输入图像路径
    :param output_path: 输出图像路径（必须包含文件名和扩展名，如：output.jpg）
    :param highpass_radius: 高频掩码的低频半径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入图像不存在: {input_path}")
    
    # 检查输出路径是否包含文件名和扩展名
    if not os.path.basename(output_path) or '.' not in os.path.basename(output_path):
        # 如果输出路径是文件夹，自动生成输出文件名
        input_filename = os.path.basename(input_path)
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(output_path, f"highpass_{name}{ext}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取图像
    image = Image.open(input_path).convert('RGB')
    
    # 转换为张量 [C, H, W]，数值范围 0~1
    transform = transforms.ToTensor()
    x = transform(image).unsqueeze(0)  # [1, 3, H, W]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    
    B, C, H, W = x.shape

    # FFT (complex tensor) → [B, C, H, W]
    f = torch.fft.fft2(x, norm="ortho")
    fshift = torch.fft.fftshift(f)

    # 生成高通掩码
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    yy = yy.to(device) - H // 2
    xx = xx.to(device) - W // 2
    mask_low = (xx**2 + yy**2 <= highpass_radius**2).float()
    mask_high = 1 - mask_low
    mask_high = mask_high[None, None, :, :]  # [1,1,H,W]

    # 应用掩码
    fshift_high = fshift * mask_high

    # IFFT
    f_ishift = torch.fft.ifftshift(fshift_high)
    img_high = torch.fft.ifft2(f_ishift, norm="ortho").real  # 取实部
    
    # 后处理：确保数值在合理范围内
    img_high = torch.clamp(img_high, 0, 1)

    highlight_mask = img_high > 0.3
    img_high = img_high + 10 * highlight_mask * (img_high - 0.7)
    img_high = torch.clamp(img_high, 0, 1)
    
    # 转换回PIL图像并保存
    output_img = transforms.ToPILImage()(img_high.squeeze(0).cpu())
    output_img.save(output_path)
    print(f"处理完成: {input_path} -> {output_path}")

# 使用示例
if __name__ == "__main__":
    input_image_path = "/root/1/img/orig_009_i1.png"
    
    # 方法1：指定完整的输出文件路径
    # output_image_path = "C:/Users/relll/Desktop/compare/in-the-wild_example/highpass_example_0.jpg"
    
    # 方法2：或者只指定输出文件夹，代码会自动生成文件名
    output_image_path = "/root/1/img"
    
    rgb_fft_image(input_image_path, output_image_path, highpass_radius=10)