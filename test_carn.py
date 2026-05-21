import torch
import numpy as np
import os
from PIL import Image
from torchsr import carn

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载 CARN 预训练模型 (×2)
    model = carn(scale=2, pretrained=True).to(device)
    model.eval()

    test_dir = "datasets/Set5"
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    psnr_list = []
    for fname in files:
        img = Image.open(os.path.join(test_dir, fname)).convert('YCbCr')
        y = np.array(img)[:, :, 0]
        h, w = y.shape
        # 下采样得到 LR
        lr = Image.fromarray(y).resize((w//2, h//2), Image.BICUBIC)
        lr = np.array(lr) / 255.0
        lr_tensor = torch.from_numpy(lr).float().permute(2,0,1).unsqueeze(0).to(device)  # CARN 需要 RGB? 不，torchsr 默认处理 RGB。但我们的数据是 Y 通道，需要转换？torchsr 内部会转换。可以直接输入灰度，但最好复制三通道。
        # 将单通道复制为3通道
        lr_tensor = lr_tensor.repeat(1,3,1,1)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        sr = sr_tensor.squeeze().cpu().numpy() * 255.0
        sr = sr.transpose(1,2,0)  # 转为 HWC
        sr = sr[:,:,0]  # 取 Y 通道
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        # 调整尺寸
        if sr.shape[0] != h or sr.shape[1] != w:
            sr = Image.fromarray(sr).resize((w, h), Image.BICUBIC)
            sr = np.array(sr)
        psnr = calc_psnr(y, sr)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")
    print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()