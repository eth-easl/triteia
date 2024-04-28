import os
import torch
import bitblas
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)
os.environ['NUMEXPR_MAX_THREADS'] = "16"

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

in_features = 4096
out_features = 4096
group_size = 4096
bitwidth = 4
original_w, linear, s, qw = bitblas.quantization.gen_quant4(
    in_features, out_features, group_size
)
zeros = torch.full((in_features // group_size, out_features), 7, dtype=torch.int32)

cuda_old_linear = CudaOldQuantLinear(
    bits=bitwidth,
    group_size=group_size,
    infeatures=in_features,
    outfeatures=out_features,
    bias=False,
)
cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)

bitblas_linear = bitblas.Linear(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype=f"uint{bitwidth}",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="quantized",  # setting for how to calculating zeros
)
# Repack weights from CudaOldQuantLinear to BitBLAS linear module
bitblas_linear.repack_from_gptq(cuda_old_linear)

# Prepare input data
m = 1  # Batch size
inp = torch.rand(m, in_features, dtype=torch.float16, device="cuda")

# Move models to CUDA for execution
cuda_old_linear = cuda_old_linear.to("cuda")
bitblas_linear = bitblas_linear.to("cuda")

# Perform inference without gradient calculations
with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)
    res_bitblas = bitblas_linear(inp)

print("CudaOldQuantLinear output:", res_cuda_old)
print("BitBLAS output:", res_bitblas)

# Verify the outputs are close within specified tolerances
torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1e-0, atol=1e-1)