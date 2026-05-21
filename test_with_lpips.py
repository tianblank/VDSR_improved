import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips

# ========== 模型定义（与训练完全一致）==========
# 请确保下面的模型定义与你训练时使用的结构完全相同。
# 这里只给出一个示例，你需要根据你的实际模型定义补充完整。
# 由于篇幅，我假设你已经有了 VDSR, DSConvLR, FSRCNN, CARN, IMDN 的定义。
# 如果你在单独的文件中定义了这些模型，可以直接导入，例如：
# from models import VDSR, DSConvLR, FSRCNN, CARN, IMDN
# 但为了独立运行，我将复制之前用过的模型定义（参考 test_carn_full.py, test_imdn_simple.py 等）。

# ---------- VDSR ----------
class VDSR(nn.Module):
    def __init__(self, num_layers=20, num_features=64):
        super(VDSR, self).__init__()
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Conv2d(1, num_features, 3, padding=1))
            elif i == num_layers-1:
                self.layers.append(nn.Conv2d(num_features, 1, 3, padding=1))
            else:
                self.layers.append(nn.Conv2d(num_features, num_features, 3, padding=1))
                self.layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        residual = x
        out = self.layers(x)
        return residual + out

# ---------- DSConv+LR ----------
class DSConvLR(nn.Module):
    def __init__(self, num_blocks=10, n_feats=64):
        super(DSConvLR, self).__init__()
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(nn.Sequential(
                nn.Conv2d(n_feats, n_feats, 3, padding=1, groups=n_feats),  # depthwise
                nn.Conv2d(n_feats, n_feats, 1),  # pointwise
                nn.ReLU(True)
            ))
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, 1, 3, padding=1)
        )
    def forward(self, x):
        out = self.head(x)
        for block in self.body:
            out = out + block(out)  # local residual
        out = self.tail(out)
        return out

# ---------- FSRCNN ----------
class FSRCNN(nn.Module):
    def __init__(self, scale=2, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        self.mid_part = []
        for _ in range(m):
            self.mid_part.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            self.mid_part.append(nn.PReLU(s))
        self.mid_part = nn.Sequential(*self.mid_part)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=scale, padding=4, output_padding=scale-1)
    def forward(self, x):
        out = self.first_part(x)
        out = self.shrink(out)
        out = self.mid_part(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out

# ---------- CARN ----------
class ResidualBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        )
    def forward(self, x):
        return x + self.body(x)

class CARN_Block(nn.Module):
    def __init__(self, n_feats, kernel_size=3):
        super(CARN_Block, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        )
        self.res = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1),
            nn.ReLU(True),
        )
    def forward(self, x):
        return x + self.body(x) + self.res(x)

class CARN(nn.Module):
    def __init__(self, scale=2, n_feats=64, num_blocks=3):
        super(CARN, self).__init__()
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.entry = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.body = nn.ModuleList([CARN_Block(n_feats) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, 3, padding=1)
        )
    def forward(self, x):
        f = self.head(x)
        x = self.entry(f)
        res = x
        for block in self.body:
            res = block(res)
        res += x
        x = self.tail(res)
        x = self.upsample(x)
        return x

# ---------- IMDN ----------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
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
        self.c2 = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, padding=1)
        self.c3 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, padding=1)
        self.act = nn.ReLU(True)
        self.ca = CALayer(in_channels, reduction=4)
    def forward(self, x):
        distilled = self.c1(x)
        distilled = self.act(distilled)
        distilled1 = distilled[:, :self.distilled_channels, :, :]
        remaining = distilled[:, self.distilled_channels:, :, :]
        distilled1 = self.c3(distilled1)
        remaining = self.c2(remaining)
        out = torch.cat([distilled1, remaining], dim=1)
        out = self.ca(out)
        return out + x

class IMDN(nn.Module):
    def __init__(self, scale=2, in_channels=1, out_channels=1, num_modules=4, nf=64):
        super(IMDN, self).__init__()
        self.fea_conv = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.act = nn.ReLU(True)
        self.IMD_modules = nn.ModuleList([IMDModule(nf) for _ in range(num_modules)])
        self.conv_last = nn.Conv2d(nf, nf, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(nf, out_channels, 3, padding=1)
        )
    def forward(self, x):
        out = self.act(self.fea_conv(x))
        for module in self.IMD_modules:
            out = module(out)
        out = self.act(self.conv_last(out))
        out = self.upsample(out)
        return out

# ========== 工具函数 ==========
def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def load_image_y(path, scale=2):
    img = Image.open(path).convert('YCbCr')
    y = np.array(img)[:, :, 0].astype(np.float32)
    h, w = y.shape
    lr_img = Image.fromarray(y.astype(np.uint8)).resize((w // scale, h // scale), Image.BICUBIC)
    lr = np.array(lr_img).astype(np.float32) / 255.0
    hr = y / 255.0
    return torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0), hr

def evaluate_model(model, device, test_dir, scale=2, lpips_fn=None):
    model.eval()
    model = model.to(device)
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list, ssim_list, lpips_list = [], [], []
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
        ssim_val = ssim(hr, sr, data_range=255)
        # LPIPS: 需要将图像转为 [0,1] 的 float32 tensor，并归一化到 [-1,1]（根据lpips库的要求）
        hr_tensor = torch.from_numpy(hr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        sr_tensor = torch.from_numpy(sr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        # 注意：lpips需要3通道图像，我们扩展为3通道（复制Y通道）
        hr_tensor = hr_tensor.repeat(1, 3, 1, 1)
        sr_tensor = sr_tensor.repeat(1, 3, 1, 1)
        lpips_val = lpips_fn(hr_tensor, sr_tensor).item()
        psnr_list.append(psnr)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        print(f"{fname}: PSNR = {psnr:.2f} dB, SSIM = {ssim_val:.4f}, LPIPS = {lpips_val:.4f}")
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    return avg_psnr, avg_ssim, avg_lpips

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vdsr', 'dsconv_lr', 'fsrcnn', 'carn', 'imdn'],
                        help='Type of model to test')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory of test images')
    parser.add_argument('--scale', type=int, default=2, help='Super-resolution scale')
    parser.add_argument('--num_blocks', type=int, default=10, help='Number of LR blocks (for dsconv_lr)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 实例化模型
    if args.model_type == 'vdsr':
        model = VDSR()
    elif args.model_type == 'dsconv_lr':
        model = DSConvLR(num_blocks=args.num_blocks)
    elif args.model_type == 'fsrcnn':
        model = FSRCNN(scale=args.scale)
    elif args.model_type == 'carn':
        model = CARN(scale=args.scale)
    elif args.model_type == 'imdn':
        model = IMDN(scale=args.scale)
    else:
        raise ValueError("Unknown model type")

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Model loaded from {args.model_path}")

    # 初始化 LPIPS 模型
    lpips_fn = lpips.LPIPS(net='alexnet').to(device)

    evaluate_model(model, device, args.test_dir, args.scale, lpips_fn)

if __name__ == "__main__":
    main()