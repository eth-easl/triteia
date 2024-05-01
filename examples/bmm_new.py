import torch

from triteia.ao.ops import native_bmm_lowprec
from triteia.ao.ops.linalg.matmul.bmm_lowprec_new import quant_bmm_248

import safetensors as st

tensors = {}
with st.safe_open(".local/4bit_gptq.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
BSZs = [1, 8, 16, 32, 64]
BITWIDTH = 4
prefix = "model.layers.0.self_attn.q_proj"
qweight = tensors[f"{prefix}.qweight"]
qzero = tensors[f"{prefix}.qzeros"]
scale = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]

bsz = 64
qweights = qweight.repeat(bsz, 1, 1)
qzeros = qzero.repeat(bsz, 1, 1)
scales = scale.repeat(bsz, 1, 1)
g_idxs = g_idx.repeat(bsz, 1)
x_dim = 4096
x = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)
bias = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)

torch.cuda.nvtx.range_push("bmm start")
output = quant_bmm_248(
    BITWIDTH,
    x,
    qweight=qweights,
    qzero=qzeros,
    scale=scales,
    g_idx=g_idxs,
    bias=bias,
)
torch.cuda.nvtx.range_pop()
print(output)