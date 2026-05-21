import torch
import torch.nn as nn

# ========== 公共组件 ==========
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class HAT(nn.Module):
    def __init__(self, channels, reduction=4):
        super(HAT, self).__init__()
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

# ========== 模型 1: VDSR ==========
class VDSR(nn.Module):
    def __init__(self, num_channels=1, base_filter=64):
        super(VDSR, self).__init__()
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

# ========== 模型 2: DSConv ==========
class DSConv(nn.Module):
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10):
        super(DSConv, self).__init__()
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

# ========== 模型 3: DSConv+HAT ==========
class DSConvHAT(nn.Module):
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10, reduction=4):
        super(DSConvHAT, self).__init__()
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

# ========== 模型 4: Eff-HASR ==========
class HASBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(HASBlock, self).__init__()
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
        super(EffHASR, self).__init__()
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

# ========== 模型 5: DSConv+LR ==========
class DSConvLR(nn.Module):
    def __init__(self, num_channels=1, base_filter=64, num_blocks=10):
        super(DSConvLR, self).__init__()
        self.conv_input = nn.Conv2d(num_channels, base_filter, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
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
            out = block(out) + out
        out = self.conv_output(out)
        out = out + residual
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ========== 模型 6: FSRCNN ==========
class FSRCNN(nn.Module):
    def __init__(self, scale=2, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.scale = scale
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )
        mid = []
        for _ in range(m):
            mid.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mid.append(nn.PReLU(s))
        self.mid_part = nn.Sequential(*mid)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )
        self.deconv = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale,
                                         padding=4, output_padding=scale-1)
        self._initialize_weights()

    def forward(self, x):
        out = self.first_part(x)
        out = self.shrink(out)
        out = self.mid_part(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
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
    elif model_name == 'fsrcnn':
        # FSRCNN 只接受 scale 和 num_channels，忽略其他参数
        scale = kwargs.get('scale', 2)
        num_channels = kwargs.get('num_channels', 1)
        return FSRCNN(scale=scale, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")