import torch
import safetensors as st
from ao.ops import native_matmul_lowprec_248

BIT_WIDTH = 4

tensors = {}

with st.safe_open(".local/quantized.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

prefix = "model.layers.0.self_attn.q_proj"

qweight = tensors[f"{prefix}.qweight"]
qzeros = tensors[f"{prefix}.qzeros"]
scales = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]

x = torch.rand((1, 320, 4096), device="cuda", dtype=torch.float16)
bias = torch.rand((1, 320, 4096), device="cuda", dtype=torch.float16)
output = native_matmul_lowprec_248(
    BIT_WIDTH,
    x,
    qweight,    
    qzeros,
    scales,
    g_idx,
    bias=bias
)