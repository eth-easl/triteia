import bitblas
import torch
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)

in_features = 1024
out_features = 1024
group_size = 128

original_w, linear, s, qw = bitblas.quantization.gen_quant4(
    in_features, out_features, group_size
)
zeros = torch.full((in_features // group_size, out_features), 7, dtype=torch.int32)

cuda_old_linear = CudaOldQuantLinear(
    bits=4,
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
    W_dtype="uint4",  # weight W dtype
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