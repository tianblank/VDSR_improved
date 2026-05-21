# train_carn_80.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
from PIL import Image

# ========== 模型定义（与之前完全一致）==========
class ResidualBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        )
    def forward(self, x):
        return x + self.body(x)

class CARN_Block(nn.Module):
    def __init__(self, n_feats, kernel_size=3):
        super(CARN_Block, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        )
        self.res = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1),
            nn.ReLU(True),
        )
    def forward(self, x):
        return x + self.body(x) + self.res(x)

class CARN(nn.Module):
    def __init__(self, scale=2, n_feats=64, num_blocks=3):
        super(CARN, self).__init__()
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.entry = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.body = nn.ModuleList([CARN_Block(n_feats) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 1, 3, padding=1)
        )
    def forward(self, x):
        f = self.head(x)
        x = self.entry(f)
        res = x
        for block in self.body:
            res = block(res)
        res += x
        x = self.tail(res)
        x = self.upsample(x)
        return x

# ========== 数据集 ==========
class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, 'r')
        self.lr = self.h5['data']
        self.hr = self.h5['label']
    def __len__(self):
        return len(self.lr)
    def __getitem__(self, idx):
        lr = self.lr[idx].astype(np.float32) / 255.0
        hr = self.hr[idx].astype(np.float32) / 255.0
        # 上采样 HR 到 82x82
        hr_pil = Image.fromarray((hr * 255).astype(np.uint8))
        hr_up = hr_pil.resize((82, 82), Image.Resampling.BICUBIC)
        hr = np.array(hr_up).astype(np.float32) / 255.0
        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(hr).unsqueeze(0)

# ========== 训练 ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集
    dataset = HDF5Dataset('/root/autodl-tmp/291_train/carn_train.h5')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(dataset)}")
    
    # 模型、损失、优化器
    model = CARN(scale=2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 尝试从已有的 checkpoint 恢复
    start_epoch = 0
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 查找最新的 CARN checkpoint
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('carn_291_epoch') and f.endswith('.pth')]
    if ckpt_files:
        # 提取 epoch 数字并排序
        epochs = [int(f.split('_epoch')[1].split('.')[0]) for f in ckpt_files]
        latest_epoch = max(epochs)
        latest_file = f'carn_291_epoch{latest_epoch}.pth'
        ckpt_path = os.path.join(checkpoint_dir, latest_file)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        start_epoch = latest_epoch
        print(f"Resumed from checkpoint: {latest_file}, epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")
    
    total_epochs = 80
    save_interval = 10
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 500 == 0:
                print(f"Epoch {epoch+1}/{total_epochs}, Iter {i}, Loss: {loss.item():.6f}")
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{total_epochs} Avg Loss: {avg_loss:.6f}")
        
        # 每 save_interval 个 epoch 保存一次
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"checkpoint/carn_291_epoch{epoch+1}.pth")
            print(f"Checkpoint saved: carn_291_epoch{epoch+1}.pth")
    
    # 保存最终模型
    torch.save(model.state_dict(), "checkpoint/carn_291_final.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()