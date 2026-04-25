# measure_flops_scale4.py
import torch
from thop import profile
from models import DSConvLR

def main():
    scale = 4
    model = DSConvLR(num_blocks=10).eval()
    # 对于 4× 模型，假设输入 patch 为 41×41 的 HR 块，则输入尺寸为 1×1×164×164
    #（因为 LR 已经通过 bicubic 放大到 HR 尺寸）
    h_in = 41 * scale  # 164
    w_in = 41 * scale  # 164
    input_tensor = torch.randn(1, 1, h_in, w_in)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"Params: {params / 1e3:.2f} K")
    print(f"FLOPs: {flops / 1e9:.3f} G")

if __name__ == "__main__":
    main()