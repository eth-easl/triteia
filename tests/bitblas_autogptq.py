import os
import torch
import safetensors as st
from fractions import Fraction
from triteia.ao.nn.linear_bitblas import Linear as BitblasLinear
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)
from triteia.ao.ops.linalg.matmul.matmul_lowprec import quant_matmul_248_bitblas
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE
os.environ['NUMEXPR_MAX_THREADS'] = "32"

prefix = "model.layers.0.self_attn.q_proj"

tensors = {}
bitwidth = 2
triton_weight = f".local/{bitwidth}bit_gptq.safetensors"
bitblas_weight = f".local/{bitwidth}bit_bitblas.safetensors"

with st.safe_open(triton_weight, framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
qweight = tensors[f"{prefix}.qweight"]
qzeros = tensors[f"{prefix}.qzeros"]
scales = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]

with st.safe_open(bitblas_weight, framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

bitblas_qweight = tensors[f"{prefix}.qweight"]
bitblas_zeros = tensors[f"{prefix}.zeros"]
bitblas_scales = tensors[f"{prefix}.scales"]
gptq_pack_factor = Fraction(bitwidth, 32)
bitblas_pack_factor = Fraction(bitwidth, 8)

in_features = qweight.shape[0] // gptq_pack_factor
out_features = qweight.shape[1]

group_size = in_features

cuda_old_linear = CudaOldQuantLinear(
    bits=bitwidth,
    group_size=group_size,
    infeatures=in_features,
    outfeatures=out_features,
    bias=False,
)
cuda_old_linear.qweight = qweight
cuda_old_linear.qzeros = qzeros
cuda_old_linear.scales = scales
cuda_old_linear.g_idx = g_idx

bitblas_linear = BitblasLinear(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype=QUANTIZED_DTYPE[bitwidth],  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="quantized",  # setting for how to calculating zeros
)
# Repack weights from CudaOldQuantLinear to BitBLAS linear module
# bitblas_linear.repack_from_gptq(cuda_old_linear)
# bitblas_linear.repack_from_weights(qweight, scales, qzeros)
bitblas_linear.qweight = bitblas_qweight
bitblas_linear.zeros = bitblas_zeros
bitblas_linear.scales = bitblas_scales
# # Prepare input data
m = 1
inp = torch.rand(m, in_features, dtype=torch.float16, device="cuda")

# Move models to CUDA for execution
cuda_old_linear = cuda_old_linear.to("cuda")
bitblas_linear = bitblas_linear.to("cuda")

# Perform inference without gradient calculations
with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)
    res_bitblas = bitblas_linear(inp)
    
torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1e-0, atol=1e-1)

assert bitblas_qweight.shape[1] == qweight.shape[0] * 4

bitblas_qweight = torch.zeros_like(bitblas_qweight)
bitblas_zeros = torch.zeros_like(bitblas_zeros)
bitblas_scales = torch.ones_like(bitblas_scales)

res_bitblas = quant_matmul_248_bitblas(
    bitwidth=bitwidth,
    x=inp,
    qweight=bitblas_qweight,
    qzero=bitblas_zeros,
    scale=bitblas_scales
)

print("CudaOldQuantLinear output:", res_cuda_old)
print("BitBLAS output:", res_bitblas)

# Verify the outputs are close within specified tolerances
