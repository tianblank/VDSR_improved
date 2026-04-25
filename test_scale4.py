# test_scale4.py (修正版)
import torch
import numpy as np
import os
import argparse
from PIL import Image
from models import DSConvLR

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="datasets/Set5")
    parser.add_argument("--scale", type=int, default=4, help="upscaling factor for bicubic resize")
    parser.add_argument("--num_blocks", type=int, default=10)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型不需要 scale 参数
    model = DSConvLR(num_blocks=opt.num_blocks).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.eval()

    files = [f for f in sorted(os.listdir(opt.test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list = []
    for fname in files:
        img = Image.open(os.path.join(opt.test_dir, fname)).convert('YCbCr')
        y = np.array(img)[:, :, 0]
        h, w = y.shape
        # 使用 bicubic 下采样再上采样作为输入（scale 决定放大倍数）
        lr = Image.fromarray(y).resize((w // opt.scale, h // opt.scale), Image.BICUBIC)
        lr = lr.resize((w, h), Image.BICUBIC)
        lr = np.array(lr)
        input_tensor = torch.from_numpy(lr / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        sr = output.squeeze().cpu().numpy() * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        psnr = calc_psnr(y, sr)
        psnr_list.append(psnr)
        print(f"{fname}: {psnr:.2f} dB")
    print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()