import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from dataset import DatasetFromHdf5
from models import get_model   # 使用统一的工厂函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=['vdsr', 'dsconv', 'dsconv_hat', 'effhasr', 'dsconv_lr', 'fsrcnn'],
                        help="模型类型")
    parser.add_argument("--datasets_path", type=str, default="datasets/train.h5")
    parser.add_argument("--weight_save_path", type=str, default="weight/model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num_blocks", type=int, default=10,
                        help="仅对 dsconv, dsconv_hat, effhasr, dsconv_lr 有效")
    parser.add_argument("--base_filter", type=int, default=64)
    parser.add_argument("--theta", type=float, default=0.01)
    parser.add_argument("--reduction", type=int, default=4,
                        help="通道注意力压缩比，仅对 dsconv_hat 和 effhasr 有效")
    parser.add_argument("--scale", type=int, default=2,
                        help="放大倍数，仅对 FSRCNN 有效")
    opt = parser.parse_args()

    os.makedirs(os.path.dirname(opt.weight_save_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"训练模型: {opt.model}")

    dataset = DatasetFromHdf5(opt.datasets_path, use_minmax_norm=True)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.num_workers, pin_memory=False)
    print(f"数据集大小: {len(dataset)}, 每轮迭代: {len(loader)}")

    # 构建参数字典
    kwargs = {
        'num_channels': 1,
        'base_filter': opt.base_filter,
    }
    if opt.model != 'vdsr':
        kwargs['num_blocks'] = opt.num_blocks
    if opt.model in ['dsconv_hat', 'effhasr']:
        kwargs['reduction'] = opt.reduction
    if opt.model == 'fsrcnn':
        kwargs['scale'] = opt.scale   # FSRCNN 需要 scale 参数

    model = get_model(opt.model, **kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.init_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)

    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{opt.epochs}, LR = {current_lr:.6f}")

        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.theta / current_lr)

            optimizer.step()
            total_loss += loss.item()

            if i % 500 == 0:
                print(f"  Iter [{i:4d}/{len(loader)}], Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{opt.weight_save_path}_epoch{epoch+1}.pth")
            print(f"  模型已保存: {opt.weight_save_path}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), opt.weight_save_path)
    print(f"\n训练完成！最终模型: {opt.weight_save_path}")

if __name__ == "__main__":
    main()