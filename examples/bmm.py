import torch

from triteia.ao.ops import native_bmm_lowprec, quant_bmm_248
import safetensors as st

tensors = {}
with st.safe_open(".local/quantized.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
BSZs = [1, 8, 16, 32, 64]
BITWIDTH = 4
prefix = "model.layers.0.self_attn.q_proj"
qweight = tensors[f"{prefix}.qweight"]
qzero = tensors[f"{prefix}.qzeros"]
scale = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]

def warmup():
    for bsz in BSZs:
        qweights = qweight.repeat(bsz, 1, 1)
        qzeros = qzero.repeat(bsz, 1, 1)
        scales = scale.repeat(bsz, 1, 1)
        g_idxs = g_idx.repeat(bsz, 1)
        for i in range(1, 15):
            x_dim = int(128 * i)
            x = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)
            bias = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)
            quant_bmm_248(
                BITWIDTH,
                x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            )
i = 15
bsz = 64
qweights = qweight.repeat(bsz, 1, 1)
qzeros = qzero.repeat(bsz, 1, 1)
scales = scale.repeat(bsz, 1, 1)
g_idxs = g_idx.repeat(bsz, 1)
x_dim = int(128 * i)
x = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)
bias = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)

torch.cuda.nvtx.range_push("bmm start")
quant_bmm_248(
    BITWIDTH,
    x,
    qweight=qweights,
    qzero=qzeros,
    scale=scales,
    g_idx=g_idxs,
    bias=bias,
)
torch.cuda.nvtx.range_pop()