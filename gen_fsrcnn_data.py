import h5py
import numpy as np
import os
from PIL import Image

def prepare_fsrcnn_data(data_path, h5_file_path, scale=2, patch_size=41, stride=21):
    hr_patch_size = patch_size * scale  # 82
    print(f"Extracting HR patches of size {hr_patch_size}x{hr_patch_size}")
    
    h5_file = h5py.File(h5_file_path, 'w')
    data_patches = []  # LR 41x41
    label_patches = []  # HR 82x82
    
    img_count = 0
    for file_name in sorted(os.listdir(data_path)):
        if not file_name.lower().endswith(('.bmp','.png','.jpg','.jpeg')):
            continue
        img_path = os.path.join(data_path, file_name)
        print(f"Processing {file_name}")
        img = Image.open(img_path).convert('YCbCr')
        y = np.array(img)[:, :, 0]
        h, w = y.shape
        if h < hr_patch_size or w < hr_patch_size:
            print(f"  Image too small, skipping")
            continue
        # 提取 HR patches
        for i in range(0, h - hr_patch_size + 1, stride):
            for j in range(0, w - hr_patch_size + 1, stride):
                hr_patch = y[i:i+hr_patch_size, j:j+hr_patch_size]
                # 下采样得到 LR patch (bicubic)
                lr_patch = Image.fromarray(hr_patch).resize((patch_size, patch_size), Image.BICUBIC)
                lr_patch = np.array(lr_patch)
                # 8 种数据增强
                for _ in range(8):
                    data_patches.append(lr_patch.copy())
                    label_patches.append(hr_patch.copy())
                    lr_patch = np.rot90(lr_patch)
                    hr_patch = np.rot90(hr_patch)
                    if _ == 3:
                        lr_patch = np.fliplr(lr_patch)
                        hr_patch = np.fliplr(hr_patch)
        img_count += 1
        print(f"  Processed, total patches now: {len(data_patches)}")
    
    print(f"\nProcessed {img_count} images")
    data_patches = np.array(data_patches, dtype=np.uint8)
    label_patches = np.array(label_patches, dtype=np.uint8)
    print(f"Data shape: {data_patches.shape}, range [{data_patches.min()},{data_patches.max()}]")
    print(f"Label shape: {label_patches.shape}, range [{label_patches.min()},{label_patches.max()}]")
    
    h5_file.create_dataset('data', data=data_patches, compression='gzip')
    h5_file.create_dataset('label', data=label_patches, compression='gzip')
    h5_file.close()
    print(f"Saved to {h5_file_path}")

if __name__ == "__main__":
    prepare_fsrcnn_data('/root/autodl-tmp/291_train_images_yuanlai', 'datasets/train_fsrcnn.h5', scale=2, patch_size=41, stride=21)