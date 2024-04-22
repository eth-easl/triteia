import os
import torch
import safetensors as st
from triteia.ao.ops.linalg.matmul.bitblas_matmul_lowprec import bitblas_quant_bmm_248

bitwidth = 4
os.environ['NUMEXPR_MAX_THREADS'] = "32"

bitblas_weight = ".local/bitblas.safetensors"
prefix = "model.layers.9.mlp.gate_proj"
tensors = {}

with st.safe_open(bitblas_weight, framework="pt", device="cuda") as f:
    for key in f.keys():
        if key.startswith(prefix):
            tensors[key] = f.get_tensor(key)

bitblas_qweight = tensors[f"{prefix}.qweight"]
bitblas_zeros = tensors[f"{prefix}.zeros"]
bitblas_scales = tensors[f"{prefix}.scales"]

in_features = bitblas_qweight.shape[1] * 2

m = 1024
inp = torch.rand(m, in_features, dtype=torch.float16, device="cuda")
    
# bitblas_qweight = torch.zeros_like(bitblas_qweight)
# bitblas_zeros = torch.zeros_like(bitblas_zeros)
# bitblas_scales = torch.zeros_like(bitblas_scales)

res_bitblas = bitblas_quant_bmm_248(
    bitwidth=bitwidth,
    x=inp,
    qweight=bitblas_qweight,
    qzero=bitblas_zeros,
    scale=bitblas_scales
)
print(res_bitblas.shape)
print("BitBLAS output:", res_bitblas)