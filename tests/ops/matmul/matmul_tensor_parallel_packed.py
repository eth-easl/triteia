import torch
from fractions import Fraction
from triteia.utils.io import read_tensors
from triteia.ao.ops.matmul.matmul_lowprec import quant_matmul_248

DEVICE = "cuda:0"
BITWIDTH = 4
gptq_tensors = read_tensors(
    ".local/tinyllama/gptq.safetensors",
    prefix="model.layers.0.self_attn",
    device=DEVICE
)
marlin_tensors = read_tensors(
    ".local/tinyllama/marlin.tp2.safetensors",
    prefix="model.layers.0.self_attn",
    device=DEVICE
)

quant_pack_factor = Fraction(4, 32)

q_proj = (gptq_tensors['q_proj.qweight'], gptq_tensors['q_proj.qzeros'], gptq_tensors['q_proj.scales'], gptq_tensors['q_proj.g_idx'])
k_proj = (gptq_tensors['k_proj.qweight'], gptq_tensors['k_proj.qzeros'], gptq_tensors['k_proj.scales'], gptq_tensors['k_proj.g_idx'])
v_proj = (gptq_tensors['v_proj.qweight'], gptq_tensors['v_proj.qzeros'], gptq_tensors['v_proj.scales'], gptq_tensors['v_proj.g_idx'])

packed_marlin_tensors = (marlin_tensors['qkv_proj.qweight'], marlin_tensors['qkv_proj.meta'])
infeatures = q_proj[0].shape[0] // quant_pack_factor

inp = torch.randn((1, infeatures), dtype=torch.float16, device="cuda:0")
q_output = quant_matmul_248(BITWIDTH, inp, *q_proj, bias=None)
k_output = quant_matmul_248(BITWIDTH, inp, *k_proj, bias=None)
v_output = quant_matmul_248(BITWIDTH, inp, *v_proj, bias=None)
