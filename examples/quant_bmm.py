import torch
import unittest
from triteia.ao.ops import bmm, native_bmm_lowprec, quant_bmm_248
from triteia.ao.ops.linalg.matmul.bmm_lowprec import loop_quant_bmm_248, bitblas_loop_quant_bmm_248
import torch.testing as tt
import safetensors as st

torch.manual_seed(0)
device = "cuda:1"

tensors = {}
with st.safe_open(
    ".local/4bit_bitblas.safetensors", framework="pt", device=device
) as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

prefix = "model.layers.0.self_attn.q_proj"
qweight = tensors[f"{prefix}.qweight"]
qzero = tensors[f"{prefix}.zeros"]
scale = tensors[f"{prefix}.scales"]
bitwidth = 4
bszs = [1]

for bsz in bszs:
    x = torch.randn((bsz, 1, 4096), device=device, dtype=torch.float16)
    # bias = torch.randn((bsz, 1, 4096), device=device, dtype=torch.float16)
    qweights = qweight.repeat(bsz, 1, 1)
    qzeros = qzero.repeat(bsz, 1, 1)
    scales = scale.repeat(bsz, 1, 1)
    qweights = qweights.to(device)
    qzeros = qzeros.to(device)
    scales = scales.to(device)
    bitblas_output = bitblas_loop_quant_bmm_248(
        bitwidth=bitwidth,
        x=x,
        qweight=qweights,
        qzero=qzeros,
        scale=scales,
        g_idx=None,
        bias=None,
    )
    print(bitblas_output)