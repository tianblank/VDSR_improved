# test_carn_robust.py
import torch
import numpy as np
import os
import requests
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

# ---------- 1. CARN 网络结构定义（与官方一致） ----------
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResidualBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bn=False):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        )
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class CARN_Block(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bn=False):
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
        res = self.body(x)
        res += self.res(x)
        res += x
        return res

class CARN(nn.Module):
    def __init__(self, scale=2, n_feats=64, num_blocks=3):
        super(CARN, self).__init__()
        self.sub_mean = MeanShift(255, [0.4488, 0.4371, 0.4040], [1.0, 1.0, 1.0], sign=-1)
        self.add_mean = MeanShift(255, [0.4488, 0.4371, 0.4040], [1.0, 1.0, 1.0], sign=1)
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.entry = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(CARN_Block(n_feats))
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )
    def forward(self, x):
        x = self.sub_mean(x)
        f = self.head(x)
        x = self.entry(f)
        res = x
        for block in self.body:
            res = block(res)
        res += x
        x = self.tail(res)
        x = self.upsample(x) + f
        x = self.add_mean(x)
        return x

# ---------- 2. 权重文件获取（自动+手动指引） ----------
def download_file(url, dest, chunk_size=8192):
    if os.path.exists(dest):
        print(f"权重文件已存在: {dest}")
        return True
    print(f"尝试从 {url} 下载 ...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        print("下载成功")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def get_carn_weight(scale=2):
    os.makedirs("weights", exist_ok=True)
    filename = f"CARN_x{scale}.pth"
    dest = os.path.join("weights", filename)
    
    # 备用下载源列表
    urls = [
        f"https://cv.snu.ac.kr/research/CARN/models/{filename}",  # 官方源
        f"https://github.com/nmhkahn/CARN-pytorch/raw/master/pretrained/{filename}",  # GitHub 镜像
        f"https://media.githubusercontent.com/media/nmhkahn/CARN-pytorch/master/pretrained/{filename}",  # 另一镜像
    ]
    for url in urls:
        if download_file(url, dest):
            return dest
    print("\n⚠️ 自动下载均失败，请手动下载权重文件：")
    print(f"   1. 访问官方发布页：https://cv.snu.ac.kr/research/CARN/")
    print(f"   2. 或直接点击链接下载：https://cv.snu.ac.kr/research/CARN/models/CARN_x{scale}.pth")
    print(f"   3. 将下载的 {filename} 放入 ./weights/ 目录")
    input("完成后按 Enter 继续...")
    if os.path.exists(dest):
        return dest
    else:
        raise FileNotFoundError(f"权重文件 {dest} 未找到")

def load_carn(scale=2, device='cuda'):
    weight_path = get_carn_weight(scale)
    model = CARN(scale=scale).to(device)
    state_dict = torch.load(weight_path, map_location='cpu')
    # 移除 'module.' 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

# ---------- 3. 测试主函数 ----------
def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def test_carn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("加载 CARN 模型...")
    model = load_carn(scale=2, device=device)
    
    test_dir = "datasets/Set5"
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list = []
    for fname in files:
        # 读取 RGB 图像（CARN 需要 RGB 输入）
        img = Image.open(os.path.join(test_dir, fname)).convert('RGB')
        # 转为 YCbCr 以便提取 Y 通道计算 PSNR
        ycbcr = img.convert('YCbCr')
        y = np.array(ycbcr)[:, :, 0]
        h, w = y.shape
        # 下采样
        lr = img.resize((w//2, h//2), Image.BICUBIC)
        lr_tensor = torch.from_numpy(np.array(lr).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr = sr_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        sr = Image.fromarray(sr).convert('YCbCr')
        sr_y = np.array(sr)[:, :, 0]
        # 确保尺寸一致
        if sr_y.shape[0] != h or sr_y.shape[1] != w:
            sr_y = Image.fromarray(sr_y).resize((w, h), Image.BICUBIC)
            sr_y = np.array(sr_y)
        psnr = calc_psnr(y, sr_y)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")
    print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    test_carn()