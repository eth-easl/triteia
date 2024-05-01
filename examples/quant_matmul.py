import os
import torch
import bitblas
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)
from triteia.ao.ops.linalg.matmul.matmul_lowprec import quant_matmul_248_bitblas

os.environ["NUMEXPR_MAX_THREADS"] = "16"

in_features = 8192 # K
group_size = in_features
out_features = 1024 # N
bitwidth = 4
device = "cuda:0"
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

cuda_old_linear = CudaOldQuantLinear(
    bits=bitwidth,
    group_size=group_size,
    infeatures=in_features,
    outfeatures=out_features,
    bias=False,
)

original_w, linear, s, qw = gen_quant(
    bitwidth, in_features, out_features, group_size
)
max_zero_int = 2*bitwidth - 1
zeros = torch.full((in_features // group_size, out_features), max_zero_int, dtype=torch.int32)

cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)

bitblas_linear = bitblas.Linear(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    A_dtype="float16",
    W_dtype=f"uint{bitwidth}",
    accum_dtype="float16",
    out_dtype="float16",
    group_size=group_size,
    with_scaling=True,
    with_zeros=True,
    zeros_mode="quantized",
    enable_tuning=False,
)
bitblas_linear = bitblas_linear.to(device)
cuda_old_linear = cuda_old_linear.to(device)
bitblas_linear.repack_from_gptq(cuda_old_linear)
m = 1  # Batch size
print(f"M: {m}, N: {out_features}, K: {in_features}")

inp = torch.rand(m, in_features, dtype=torch.float16, device=device)
res_bitblas = quant_matmul_248_bitblas(
    bitwidth,
    inp,
    bitblas_linear.qweight,
    bitblas_linear.zeros.T,
    bitblas_linear.scales.cuda(),
    None
)

with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)

print(f"CudaOldQuantLinear output: {res_cuda_old}")
print(f"BitBLAS output: {res_bitblas}")
torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1, atol=10)