# train_imdn_80.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
from PIL import Image

# ========== 模型定义 ==========
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=4):
        super(IMDModule, self).__init__()
        self.distilled_channels = in_channels // distillation_rate
        self.remaining_channels = in_channels - self.distilled_channels
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.c2 = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, padding=1)
        self.c3 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, padding=1)
        self.act = nn.ReLU(True)
        self.ca = CALayer(in_channels, reduction=4)
    def forward(self, x):
        distilled = self.c1(x)
        distilled = self.act(distilled)
        distilled1 = distilled[:, :self.distilled_channels, :, :]
        remaining = distilled[:, self.distilled_channels:, :, :]
        distilled1 = self.c3(distilled1)
        remaining = self.c2(remaining)
        out = torch.cat([distilled1, remaining], dim=1)
        out = self.ca(out)
        return out + x

class IMDN(nn.Module):
    def __init__(self, scale=2, in_channels=1, out_channels=1, num_modules=4, nf=64):
        super(IMDN, self).__init__()
        self.fea_conv = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.act = nn.ReLU(True)
        self.IMD_modules = nn.ModuleList([IMDModule(nf) for _ in range(num_modules)])
        self.conv_last = nn.Conv2d(nf, nf, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(nf, out_channels, 3, padding=1)
        )
    def forward(self, x):
        out = self.act(self.fea_conv(x))
        for module in self.IMD_modules:
            out = module(out)
        out = self.act(self.conv_last(out))
        out = self.upsample(out)
        return out

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
        hr_pil = Image.fromarray((hr * 255).astype(np.uint8))
        hr_up = hr_pil.resize((82, 82), Image.Resampling.BICUBIC)
        hr = np.array(hr_up).astype(np.float32) / 255.0
        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(hr).unsqueeze(0)

# ========== 训练 ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = HDF5Dataset('/root/autodl-tmp/291_train/carn_train.h5')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(dataset)}")
    
    model = IMDN(scale=2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    start_epoch = 0
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 查找最新的 IMDN checkpoint
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('imdn_291_epoch') and f.endswith('.pth')]
    if ckpt_files:
        epochs = [int(f.split('_epoch')[1].split('.')[0]) for f in ckpt_files]
        latest_epoch = max(epochs)
        latest_file = f'imdn_291_epoch{latest_epoch}.pth'
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
        
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"checkpoint/imdn_291_epoch{epoch+1}.pth")
            print(f"Checkpoint saved: imdn_291_epoch{epoch+1}.pth")
    
    torch.save(model.state_dict(), "checkpoint/imdn_291_final.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()