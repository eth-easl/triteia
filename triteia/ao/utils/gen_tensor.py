import torch
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)

def gen_quant(bitwidth, k, n, groupsize=-1):
    maxq = 2**bitwidth
    w = torch.randn((k, n), dtype=torch.half, device="cpu")
    original_w = w.clone()
    if groupsize == -1:
        groupsize = k
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    # Quantize.
    w = torch.round(w / s).int()
    # Unsigned storage.
    w += (maxq) // 2
    w = torch.clamp(w, 0, maxq)
    # Dequantize.
    ref = (w - (maxq) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()
    return original_w, linear, s, (w - (maxq) // 2)

def generate_quantized_weight(bitwidth, in_features, out_features, group_size):
    original_w, linear, s, qw = gen_quant(
        bitwidth, in_features, out_features, group_size
    )
    cuda_old_linear = CudaOldQuantLinear(
        bits=bitwidth,
        group_size=in_features,
        infeatures=in_features,
        outfeatures=out_features,
        bias=False,
    )
    max_zero_int = 2*bitwidth - 1
    zeros = torch.full((in_features // group_size, out_features), max_zero_int, dtype=torch.int32)
    cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)
    return cuda_old_linear.qweight, cuda_old_linear.scales, cuda_old_linear.qzeros