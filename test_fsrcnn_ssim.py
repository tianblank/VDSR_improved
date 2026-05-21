import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ========== 原版 FSRCNN 模型定义（与训练一致）==========
class FSRCNN(nn.Module):
    def __init__(self, scale=2, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        # 特征提取层
        self.first_part = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        # 收缩层
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        # 映射层（m个 3x3 卷积）
        self.mid_part = []
        for _ in range(m):
            self.mid_part.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            self.mid_part.append(nn.PReLU(s))
        self.mid_part = nn.Sequential(*self.mid_part)
        # 扩张层
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        # 反卷积上采样
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=scale, padding=4, output_padding=scale-1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.shrink(out)
        out = self.mid_part(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out

# ========== 工具函数 ==========
def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def calc_ssim(img1, img2):
    return ssim(img1, img2, data_range=255)

def load_image_y(path, scale=2):
    img = Image.open(path).convert('YCbCr')
    y = np.array(img)[:, :, 0].astype(np.float32)
    h, w = y.shape
    lr_img = Image.fromarray(y.astype(np.uint8)).resize((w // scale, h // scale), Image.BICUBIC)
    lr = np.array(lr_img).astype(np.float32) / 255.0
    hr = y / 255.0
    return torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0), hr

def evaluate_model(model, device, test_dir, scale=2):
    model.eval()
    model = model.to(device)
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list = []
    ssim_list = []
    for fname in files:
        lr_tensor, hr_np = load_image_y(os.path.join(test_dir, fname), scale)
        lr_tensor = lr_tensor.to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr = sr_tensor.squeeze().cpu().numpy() * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        hr = (hr_np * 255).astype(np.uint8)
        if sr.shape != hr.shape:
            sr = np.array(Image.fromarray(sr).resize((hr.shape[1], hr.shape[0]), Image.BICUBIC))
        psnr = calc_psnr(hr, sr)
        ssim_val = calc_ssim(hr, sr)
        psnr_list.append(psnr)
        ssim_list.append(ssim_val)
        print(f"{fname}: PSNR = {psnr:.2f} dB, SSIM = {ssim_val:.4f}")
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--d', type=int, default=56, help='Number of channels in first part')
    parser.add_argument('--s', type=int, default=12, help='Number of channels in mapping part')
    parser.add_argument('--m', type=int, default=4, help='Number of mapping layers')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FSRCNN(scale=args.scale, d=args.d, s=args.s, m=args.m).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Model loaded from {args.model_path}")

    evaluate_model(model, device, args.test_dir, args.scale)

if __name__ == "__main__":
    main()