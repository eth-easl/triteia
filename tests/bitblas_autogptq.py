import bitblas
import torch
import safetensors as st
from triteia.ao.ops.nn.linear_bitblas import Linear as BitblasLinear
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)

prefix = "model.layers.0.self_attn.q_proj"

tensors = {}
triton_weight = ".local/quantized.safetensors"
bitblas_weight = ".local/bitblas.safetensors"
bitwidth = 4

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

in_features = qweight.shape[0] * 32 // bitwidth
out_features = qweight.shape[1]
group_size = in_features

cuda_old_linear = CudaOldQuantLinear(
    bits=4,
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
# bitblas_linear.repack_from_gptq(cuda_old_linear)
# bitblas_linear.repack_from_weights(qweight, scales, qzeros)
bitblas_linear.qweight = bitblas_qweight
bitblas_linear.zeros = bitblas_zeros
bitblas_linear.scales = bitblas_scales
# # Prepare input data
m = 2048
inp = torch.rand(m, in_features, dtype=torch.float16, device="cuda")

# Move models to CUDA for execution
cuda_old_linear = cuda_old_linear.to("cuda")
bitblas_linear = bitblas_linear.to("cuda")

# Perform inference without gradient calculations
with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)
    res_bitblas = bitblas_linear(inp)
torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1e-0, atol=1e-1)

M = inp.shape[0]
N = inp.shape[1]
K = out_features
matmul_config = bitblas.MatmulConfig(
    M=M,
    N=N,
    K=K,
    A_dtype="float16",
    W_dtype="uint4",
    accum_dtype="float16",
    out_dtype="float16",
    with_bias=False,
    group_size=qweight.shape[0] * 8,
    with_scaling=True,
    with_zeros=True,
    zeros_mode="quantized",
)
matmul = bitblas.Matmul(config=matmul_config)
print("CudaOldQuantLinear output:", res_cuda_old)
print("BitBLAS output:", res_bitblas)
print(
    "BitBLAS matmul output: ",
    matmul(inp, bitblas_qweight, zeros=bitblas_zeros, scale=bitblas_scales),
)
# Verify the outputs are close within specified tolerances
