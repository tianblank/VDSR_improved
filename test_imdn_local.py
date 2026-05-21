# test_imdn_local.py
import torch
import numpy as np
import os
from PIL import Image
from collections import OrderedDict
import importlib.util

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def get_imdn_model(scale=2, weight_path="weights/IMDN_x2.pth"):
    # 下载或加载 architecture.py
    arch_path = "architecture.py"
    if not os.path.exists(arch_path):
        import requests
        url = "https://raw.githubusercontent.com/Zheng222/IMDN/master/model/architecture.py"
        r = requests.get(url)
        with open(arch_path, 'w') as f:
            f.write(r.text)
    spec = importlib.util.spec_from_file_location("architecture", arch_path)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    IMDN_class = arch.IMDN
    model = IMDN_class(upscale_factor=scale)
    state = torch.load(weight_path, map_location='cpu')
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_imdn_model(scale=2, weight_path="weights/IMDN_x2.pth").to(device)
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
        # IMDN 需要三通道
        lr_tensor = lr_tensor.repeat(1,3,1,1)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr = sr_tensor.squeeze().cpu().numpy() * 255.0
        sr = sr.transpose(1,2,0)
        sr = sr[:,:,0]
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        if sr.shape[0] != h or sr.shape[1] != w:
            sr = Image.fromarray(sr).resize((w, h), Image.BICUBIC)
            sr = np.array(sr)
        psnr = calc_psnr(y, sr)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")
    print(f"\nAverage PSNR on Set5: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()