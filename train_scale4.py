# train_scale4.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from dataset import DatasetFromHdf5
from models import DSConvLR   # 假设您的模型类名为 DSConvLR，且支持 scale 参数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", type=str, default="datasets/train_scale4.h5")
    parser.add_argument("--weight_save_path", type=str, default="weight/dsconv_lr_scale4.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--scale", type=int, default=4, help="upscaling factor")
    opt = parser.parse_args()

    os.makedirs(os.path.dirname(opt.weight_save_path), exist_ok=True)

    # 数据集（注意：需要确保 DatasetFromHdf5 能够根据 scale 生成 LR patch？实际上训练数据已经固定好 LR 和 HR 的缩放关系，patch 提取时已经生成。所以这里不需要再传 scale，只需正常加载）
    dataset = DatasetFromHdf5(opt.datasets_path)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    print(f"数据集大小: {len(dataset)}")
    print(f"每轮迭代: {len(loader)}")
    print(f"放大倍数: {opt.scale}")

    model = DSConvLR(num_blocks=opt.num_blocks).cuda()     # 假设模型接受 scale
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.init_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.1)

    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{opt.epochs}, LR = {current_lr}")

        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01/current_lr)
            optimizer.step()
            total_loss += loss.item()
            if i % 500 == 0:
                print(f"  Iter [{i}/{len(loader)}], Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        scheduler.step()

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"{opt.weight_save_path}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), opt.weight_save_path)
    print("训练完成！")

if __name__ == "__main__":
    main()