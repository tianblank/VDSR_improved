import torch
import numpy as np
import os
import argparse
from PIL import Image
from models import get_model

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=['vdsr', 'dsconv', 'dsconv_hat', 'effhasr'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="datasets/Set5")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--base_filter", type=int, default=64)
    parser.add_argument("--reduction", type=int, default=4,
                        help="通道注意力压缩比，需与训练时一致")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"测试模型: {opt.model}")

    kwargs = {
        'num_channels': 1,
        'base_filter': opt.base_filter,
    }
    if opt.model != 'vdsr':
        kwargs['num_blocks'] = opt.num_blocks
    if opt.model in ['dsconv_hat', 'effhasr']:
        kwargs['reduction'] = opt.reduction

    model = get_model(opt.model, **kwargs).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.eval()

    files = [f for f in sorted(os.listdir(opt.test_dir))
             if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    if not files:
        print(f"错误：{opt.test_dir} 中没有图像文件")
        return

    psnr_list = []
    for fname in files:
        img = Image.open(os.path.join(opt.test_dir, fname)).convert('YCbCr')
        y = np.array(img)[:, :, 0]
        h, w = y.shape
        # Bicubic 下采样再上采样
        lr_img = Image.fromarray(y).resize((w // opt.scale, h // opt.scale), Image.BICUBIC)
        lr_img = lr_img.resize((w, h), Image.BICUBIC)
        lr = np.array(lr_img)

        lr_tensor = torch.from_numpy(lr / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr = sr_tensor.squeeze().cpu().numpy() * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)

        psnr = calc_psnr(y, sr)
        psnr_list.append(psnr)
        print(f"{fname}: {psnr:.2f} dB")

    print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()