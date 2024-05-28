import torch
import safetensors as st
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.lib.marlin import Layer_2_4 as MarlinLayer, mask_creator
from triteia.ao.ops.matmul.native_mm_lowprec import native_matmul_lowprec_248
from marlin_2_4 import quant_4_nt

prefix = "model.layers.0.mlp.down_proj"
gptq_tensors_file = ".local/tinyllama/model.safetensors"
marlin_tensors = {}
gptq_tensors = {}
DEV = "cuda:0"

with st.safe_open(
    gptq_tensors_file, framework="pt", device=DEV
) as f:
    for key in f.keys():
        if key.startswith(prefix):
            module_name = key.removeprefix(prefix + ".")
            gptq_tensors[module_name] = f.get_tensor(key)

dequantized_weight = dequantize_weight(
    gptq_tensors["qweight"],
    gptq_tensors["qzeros"],
    gptq_tensors["scales"],
).to(torch.float16)
scales = gptq_tensors["scales"]

m, k = dequantized_weight.shape[0], dequantized_weight.shape[1]
n_input = 1
input_dim = k
####

x = torch.randn((n_input, input_dim), dtype=torch.half, device=DEV)
ref_output = native_matmul_lowprec_248(
    4, x, 
    gptq_tensors['qweight'],
    gptq_tensors['qzeros'],
    gptq_tensors['scales'],
    gptq_tensors['g_idx'],
    bias=None
)
print(ref_output)
print(f"m: {m}, k: {k}")
layer = MarlinLayer(
    infeatures=k,
    outfeatures=m,
    groupsize=-1
)
print(f"w: {dequantized_weight.t().shape}, scales: {scales.shape}")

layer.n = m
layer.k = k
layer.groupsize = k
layer.B = torch.empty((k // 32, m * 2), dtype=torch.int, device=DEV)
layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
layer.s = torch.empty((1, m), dtype=torch.half, device=DEV)
layer.pack(
    dequantized_weight.t(),
    scales=scales,
    trans=True,
)
output = layer(x)
print(output)