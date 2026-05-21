# test_imdn_robust.py
import torch
import numpy as np
import os
import requests
from PIL import Image
from collections import OrderedDict
import importlib.util

def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"下载 {url} ...")
    r = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def get_imdn(scale=2, device='cuda'):
    # 下载模型定义文件
    arch_url = "https://raw.githubusercontent.com/Zheng222/IMDN/master/model/architecture.py"
    arch_path = "architecture.py"
    if not os.path.exists(arch_path):
        download_file(arch_url, arch_path)
    
    spec = importlib.util.spec_from_file_location("architecture", arch_path)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    IMDN_class = arch.IMDN
    
    # 下载权重
    weight_path = f"weights/IMDN_x{scale}.pth"
    os.makedirs("weights", exist_ok=True)
    if not os.path.exists(weight_path):
        url = f"https://github.com/Zheng222/IMDN/raw/master/checkpoints/IMDN_x{scale}.pth"
        download_file(url, weight_path)
        # 如果失败，提示手动下载
        if not os.path.exists(weight_path):
            print(f"自动下载失败，请手动下载 {url} 到 {weight_path}")
            input("按 Enter 继续...")
    
    model = IMDN_class(upscale_factor=scale).to(device)
    state = torch.load(weight_path, map_location='cpu')
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def test_imdn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_imdn(scale=2, device=device)
    test_dir = "datasets/Set5"
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list = []
    for fname in files:
        img = Image.open(os.path.join(test_dir, fname)).convert('YCbCr')
        y = np.array(img)[:, :, 0]
        h, w = y.shape
        lr = Image.fromarray(y).resize((w//2, h//2), Image.BICUBIC)
        lr = np.array(lr) / 255.0
        lr_tensor = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0).to(device)
        # IMDN 需要 RGB 三通道
        lr_tensor = lr_tensor.repeat(1,3,1,1)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr = sr_tensor.squeeze().cpu().numpy() * 255.0
        sr = sr.transpose(1,2,0)
        sr = sr[:,:,0]  # Y 通道
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        if sr.shape[0] != h or sr.shape[1] != w:
            sr = Image.fromarray(sr).resize((w, h), Image.BICUBIC)
            sr = np.array(sr)
        psnr = calc_psnr(y, sr)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")
    print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    test_imdn()