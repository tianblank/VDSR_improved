# measure_flops_scale2.py
import torch
from thop import profile
from models import DSConvLR

def main():
    model = DSConvLR(num_blocks=10).eval()
    # 对于 2× 模型，输入尺寸应该为放大后的 HR 大小：41*2 = 82
    # 因为模型接收的是 bicubic 上采样后的 LR（与 HR 同尺寸）
    input_size = 41 * 2   # 82
    input_tensor = torch.randn(1, 1, input_size, input_size)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"Params: {params / 1e3:.2f} K")
    print(f"FLOPs: {flops / 1e9:.3f} G")

if __name__ == "__main__":
    main()