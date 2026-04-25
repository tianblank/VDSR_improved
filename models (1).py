import torch
import torch.nn as nn

# ========== 公共组件 ==========
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class HAT(nn.Module):
    """轻量级通道注意力模块 (保留，但新模型不用)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ========== 模型 1: 原始 VDSR ==========
class VDSR(nn.Module):
    def __init__(self, num_channels=1, base_filter=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, base_filter, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        middle = []
        for _ in range(18):
            middle.append(nn.Conv2d(base_filter, base_filter, 3, padding=1))
            middle.append(nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*middle)
        self.conv_last = nn.Conv2d(base_filter, num_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.middle(out)
        out = self.conv_last(out)
        out = out + residual
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ========== 模型 2: 仅深度可分离卷积 (无注意力，无局部残差) ==========
class DSConv(nn.Module):
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10):
        super().__init__()
        self.conv_input = nn.Conv2d(num_channels, base_filter, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(DepthwiseSeparableConv(base_filter, base_filter))
            blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
        self.conv_output = nn.Conv2d(base_filter, num_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv_input(x))
        out = self.blocks(out)
        out = self.conv_output(out)
        out = out + residual
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ========== 模型 3: DSConv + 通道注意力 (无局部残差) ==========
class DSConvHAT(nn.Module):
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10, reduction=4):
        super().__init__()
        self.conv_input = nn.Conv2d(num_channels, base_filter, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(DepthwiseSeparableConv(base_filter, base_filter))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(HAT(base_filter, reduction=reduction))
        self.blocks = nn.Sequential(*blocks)
        self.conv_output = nn.Conv2d(base_filter, num_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv_input(x))
        out = self.blocks(out)
        out = self.conv_output(out)
        out = out + residual
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ========== 模型 4: 完整 Eff-HASR (含注意力 + 局部残差) ==========
class HASBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(channels, channels)
        self.act = nn.ReLU(inplace=True)
        self.hat = HAT(channels, reduction=reduction)

    def forward(self, x):
        identity = x
        out = self.dw_conv(x)
        out = self.act(out)
        out = self.hat(out)
        out = out + identity
        return out

class EffHASR(nn.Module):
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10, reduction=4):
        super().__init__()
        self.conv_input = nn.Conv2d(num_channels, base_filter, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        blocks = [HASBlock(base_filter, reduction) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.conv_output = nn.Conv2d(base_filter, num_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv_input(x))
        out = self.blocks(out)
        out = self.conv_output(out)
        out = out + residual
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ========== 新模型 5: 深度可分离卷积 + 局部残差 (无注意力) ==========
class DSConvLR(nn.Module):
    """深度可分离卷积 + 局部残差 (无 HAT)"""
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10):
        super().__init__()
        self.conv_input = nn.Conv2d(num_channels, base_filter, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 使用 ModuleList 以便在每个块内部添加局部残差
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    DepthwiseSeparableConv(base_filter, base_filter),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv_output = nn.Conv2d(base_filter, num_channels, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv_input(x))
        for block in self.blocks:
            out = block(out) + out  # 局部残差连接
        out = self.conv_output(out)
        out = out + residual       # 全局残差连接
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ========== 工厂函数 ==========
def get_model(model_name, **kwargs):
    if model_name == 'vdsr':
        return VDSR(**kwargs)
    elif model_name == 'dsconv':
        return DSConv(**kwargs)
    elif model_name == 'dsconv_hat':
        return DSConvHAT(**kwargs)
    elif model_name == 'effhasr':
        return EffHASR(**kwargs)
    elif model_name == 'dsconv_lr':
        return DSConvLR(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")