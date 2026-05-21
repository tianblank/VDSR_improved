# test_imdn_complete.py
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from collections import OrderedDict

# -------------------- IMDN 网络定义 --------------------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=4):
        super(IMDModule, self).__init__()
        self.distilled_channels = in_channels // distillation_rate
        self.remaining_channels = in_channels - self.distilled_channels
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, padding=1)
        self.c3 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.ca = CALayer(in_channels, reduction=4)

    def forward(self, x):
        distilled = self.c1(x)
        distilled = self.act(distilled)
        distilled1 = distilled[:, :self.distilled_channels, :, :]
        distilled1 = self.c3(distilled1)
        remaining = distilled[:, self.distilled_channels:, :, :]
        remaining = self.c2(remaining)
        out = torch.cat([distilled1, remaining], dim=1)
        out = self.ca(out)
        return out + x

class IMDN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=1, out_channels=1, num_modules=4):
        super(IMDN, self).__init__()
        self.fea_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.IMD_modules = nn.ModuleList([IMDModule(64) for _ in range(num_modules)])
        self.conv_last = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.fea_conv(x))
        for layer in self.IMD_modules:
            out = layer(out)
        out = self.act(self.conv_last(out))
        out = self.upsample(out)
        return out

# -------------------- 工具函数 --------------------
def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def load_imdn_weights(model, weight_path):
    state = torch.load(weight_path, map_location='cpu')
    # 移除 'module.' 前缀
    new_state = OrderedDict()
    for k, v in state.items():
        name = k.replace('module.', '')
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading IMDN model...")
    model = IMDN(upscale_factor=2, in_channels=3, out_channels=3).to(device)  # IMDN 内部需要3通道
    weight_path = "weights/IMDN_x2.pth"
    if not os.path.exists(weight_path):
        print(f"权重文件 {weight_path} 不存在，请先下载。")
        return
    model = load_imdn_weights(model, weight_path)
    model.eval()
    
    test_dir = "datasets_bak/Set5"
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    if not files:
        print(f"测试目录 {test_dir} 中没有图像。")
        return

    psnr_list = []
    for fname in files:
        # 读取 RGB 图像（IMDN 需要 RGB 输入）
        img_rgb = Image.open(os.path.join(test_dir, fname)).convert('RGB')
        # 获取 Y 通道用于 PSNR 计算
        img_ycbcr = img_rgb.convert('YCbCr')
        y = np.array(img_ycbcr)[:, :, 0]
        h, w = y.shape

        # 下采样得到 LR
        lr_rgb = img_rgb.resize((w//2, h//2), Image.BICUBIC)
        lr_tensor = torch.from_numpy(np.array(lr_rgb).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        sr_rgb = sr_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255.0
        sr_rgb = np.clip(sr_rgb, 0, 255).astype(np.uint8)
        sr_ycbcr = Image.fromarray(sr_rgb).convert('YCbCr')
        sr_y = np.array(sr_ycbcr)[:, :, 0]
        # 尺寸对齐
        if sr_y.shape[0] != h or sr_y.shape[1] != w:
            sr_y = Image.fromarray(sr_y).resize((w, h), Image.BICUBIC)
            sr_y = np.array(sr_y)
        psnr = calc_psnr(y, sr_y)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")
    print(f"\nAverage PSNR on Set5: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()