import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class DatasetFromHdf5(Dataset):
    def __init__(self, h5_file_path, use_minmax_norm=True):
        super(DatasetFromHdf5, self).__init__()
        print("正在将数据加载到内存中...")
        with h5py.File(h5_file_path, 'r') as f:
            data = f['data'][:]   # (N, H, W)
            target = f['label'][:]
        
        # 转换为 float32
        data = data.astype(np.float32)
        target = target.astype(np.float32)
        
        if use_minmax_norm:
            # 计算全局最小最大值
            data_min = data.min()
            data_max = data.max()
            print(f"原始数据范围: [{data_min}, {data_max}]")
            # 拉伸到 [0,1]
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
                target = (target - data_min) / (data_max - data_min)
            else:
                data = data - data_min
                target = target - data_min
        else:
            # 固定除以 255（假设原始数据是 0-255）
            data = data / 255.0
            target = target / 255.0
        
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        print(f"归一化后范围: [{self.data.min():.3f}, {self.data.max():.3f}]")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.target[idx].unsqueeze(0)