import os
import torch
import safetensors as st
from triteia.ao.ops import native_matmul_lowprec_248, quant_matmul_248
from triteia.ao.ops.linalg.matmul.bitblas_matmul_lowprec import bitblas_quant_bmm_248

os.environ["NUMEXPR_MAX_THREADS"] = "16"

BIT_WIDTH = 4

triton_weight = ".local/quantized.safetensors"
bitblas_weight = ".local/bitblas.safetensors"
prefix = "model.layers.0.self_attn.q_proj"
tensors = {}

with st.safe_open(triton_weight, framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

qweight = tensors[f"{prefix}.qweight"]
qzeros = tensors[f"{prefix}.qzeros"]
scales = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]

x = torch.rand((320, 4096), device="cuda", dtype=torch.float16)
bias = torch.rand((320, 4096), device="cuda", dtype=torch.float16)

output = native_matmul_lowprec_248(
    BIT_WIDTH,
    x,
    qweight,
    qzeros,
    scales,
    g_idx,
)
print("native output")
print(output)
with st.safe_open(bitblas_weight, framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

bitblas_qweight = tensors[f"{prefix}.qweight"]
bitblas_qzeros = tensors[f"{prefix}.zeros"]
bitblas_scales = tensors[f"{prefix}.scales"]

output_bitblas = bitblas_quant_bmm_248(
    BIT_WIDTH,
    x,
    qweight=bitblas_qweight,
    qzero=bitblas_qzeros,
    scale=bitblas_scales,
)
print("bitblas output")
print(output_bitblas)
