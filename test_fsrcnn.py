# test_fsrcnn.py
import torch
import numpy as np
import os
import argparse
from PIL import Image
from models import get_model   # 确保 models.py 中有 FSRCNN 类

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="datasets/Set5")
    parser.add_argument("--scale", type=int, default=2)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 FSRCNN 模型
    model = get_model('fsrcnn', scale=opt.scale, num_channels=1).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.eval()

    files = [f for f in sorted(os.listdir(opt.test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list = []
    for fname in files:
        # 读取 Y 通道
        img = Image.open(os.path.join(opt.test_dir, fname)).convert('YCbCr')
        y = np.array(img)[:, :, 0]
        h, w = y.shape
        # 下采样得到 LR (尺寸为 HR/scale)
        lr = Image.fromarray(y).resize((w // opt.scale, h // opt.scale), Image.BICUBIC)
        lr = np.array(lr) / 255.0
        input_tensor = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        sr = output.squeeze().cpu().numpy() * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        # 调整尺寸到与原图一致
        if sr.shape[0] != h or sr.shape[1] != w:
            sr = Image.fromarray(sr).resize((w, h), Image.BICUBIC)
            sr = np.array(sr)
        psnr = calc_psnr(y, sr)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")
    print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()