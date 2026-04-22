import h5py
import numpy as np
import os
from PIL import Image

def prepare_train_data(data_path, h5_file_path, scale=2, patch_size=41, stride=21):
    print(f"开始处理训练数据...")
    print(f"数据源路径: {data_path}")
    print(f"输出文件: {h5_file_path}")
    print(f"缩放因子: {scale}, patch大小: {patch_size}, 步长: {stride}")
    
    h5_file = h5py.File(h5_file_path, 'w')
    data_patches = []
    label_patches = []
    img_count = 0
    
    for file_name in sorted(os.listdir(data_path)):
        if not file_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(data_path, file_name)
        print(f"处理图像: {file_name}")
        
        try:
            # 读取原图，转为YCbCr，取Y通道
            img = Image.open(img_path).convert('YCbCr')
            img = np.array(img)
            if len(img.shape) == 3:
                img = img[:, :, 0]  # 只保留Y通道
            
            h, w = img.shape
            if h < patch_size or w < patch_size:
                print(f"  图像太小，跳过")
                continue
            
            # 生成低分辨率图像（bicubic下采样再上采样）
            img_pil = Image.fromarray(img)
            lr_pil = img_pil.resize((w // scale, h // scale), Image.BICUBIC)
            lr_pil = lr_pil.resize((w, h), Image.BICUBIC)
            img_lr = np.array(lr_pil)
            
            # 尺寸对齐（避免因取整导致的不一致）
            if img.shape != img_lr.shape:
                min_h = min(img.shape[0], img_lr.shape[0])
                min_w = min(img.shape[1], img_lr.shape[1])
                img = img[:min_h, :min_w]
                img_lr = img_lr[:min_h, :min_w]
            
            h, w = img.shape
            print(f"  尺寸: {h}x{w}, 原始范围: [{img.min()}, {img.max()}]")
            
            # 提取patches
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    label_patch = img[i:i+patch_size, j:j+patch_size]
                    data_patch = img_lr[i:i+patch_size, j:j+patch_size]
                    
                    # 8种数据增强（旋转+翻转）
                    for k in range(8):
                        data_patches.append(data_patch.copy())
                        label_patches.append(label_patch.copy())
                        data_patch = np.rot90(data_patch)
                        label_patch = np.rot90(label_patch)
                        if k == 3:
                            data_patch = np.fliplr(data_patch)
                            label_patch = np.fliplr(label_patch)
            
            img_count += 1
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    print(f"\n总共处理了 {img_count} 张图像")
    
    if len(data_patches) == 0:
        print("错误：没有生成任何patches")
        h5_file.close()
        return
    
    # 转换为numpy数组（uint8）
    data_patches = np.array(data_patches, dtype=np.float32)
    label_patches = np.array(label_patches, dtype=np.float32)
    
    print(f"拉伸前数据范围: [{data_patches.min()}, {data_patches.max()}]")
    print(f"拉伸前标签范围: [{label_patches.min()}, {label_patches.max()}]")
    
    # ========== 关键：拉伸到完整的 [0, 255] ==========
    global_min = min(data_patches.min(), label_patches.min())
    global_max = max(data_patches.max(), label_patches.max())
    
    if global_max - global_min > 0:
        data_patches = (data_patches - global_min) / (global_max - global_min) * 255.0
        label_patches = (label_patches - global_min) / (global_max - global_min) * 255.0
    else:
        data_patches = data_patches - global_min  # 已经是常数
        label_patches = label_patches - global_min
    
    # 转换为 uint8
    data_patches = data_patches.astype(np.uint8)
    label_patches = label_patches.astype(np.uint8)
    
    print(f"拉伸后数据范围: [{data_patches.min()}, {data_patches.max()}]")
    print(f"拉伸后标签范围: [{label_patches.min()}, {label_patches.max()}]")
    # ================================================
    
    # 保存到HDF5（使用压缩减小体积）
    h5_file.create_dataset('data', data=data_patches, compression='gzip', compression_opts=4)
    h5_file.create_dataset('label', data=label_patches, compression='gzip', compression_opts=4)
    h5_file.close()
    
    file_size = os.path.getsize(h5_file_path) / (1024**3)
    print(f"\n完成！共生成 {len(data_patches)} 个训练patches")
    print(f"HDF5文件保存在: {h5_file_path}")
    print(f"文件大小: {file_size:.2f} GB")

if __name__ == "__main__":
    # 请根据实际路径修改
    prepare_train_data(
        data_path='datasets/291',      # 原始图像文件夹
        h5_file_path='datasets/train_x4.h5',  # 输出文件
        scale=4,
        patch_size=41,
        stride=21
    )