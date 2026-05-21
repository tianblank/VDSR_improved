import torch
import numpy as np
import os
from PIL import Image
from super_image import CarnModel

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CARN model via super-image...")
    # 库会自动从 Hugging Face Hub 下载模型缓存到本地
    model = CarnModel.from_pretrained('eugenesiow/carn', scale=2).to(device)
    model.eval()

    test_dir = "datasets/Set5"
    files = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(('.png','.jpg','.bmp'))]
    if not files:
        print(f"No images found in {test_dir}")
        return

    psnr_list = []
    for fname in files:
        img_path = os.path.join(test_dir, fname)
        # 读取 RGB 图像
        img = Image.open(img_path).convert('RGB')
        # 获取 Y 通道用于计算 PSNR
        ycbcr = img.convert('YCbCr')
        y = np.array(ycbcr)[:, :, 0]
        h, w = y.shape

        # 转换为 tensor 并归一化
        lr_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # 处理输出结果
        sr = sr_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255.0
        sr = np.clip(sr, 0, 255).astype(np.uint8)
        sr = Image.fromarray(sr).convert('YCbCr')
        sr_y = np.array(sr)[:, :, 0]

        # 确保尺寸对齐
        if sr_y.shape[0] != h or sr_y.shape[1] != w:
            sr_y = Image.fromarray(sr_y).resize((w, h), Image.BICUBIC)
            sr_y = np.array(sr_y)

        psnr = calc_psnr(y, sr_y)
        psnr_list.append(psnr)
        print(f"{fname}: PSNR = {psnr:.2f} dB")

    print(f"\nAverage PSNR on Set5: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()