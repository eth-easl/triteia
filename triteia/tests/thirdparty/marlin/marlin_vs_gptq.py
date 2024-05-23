import os
import torch
import torch.nn as nn
import safetensors as st
import triteia.lib.marlin as marlin
from triteia.ao.ops.matmul.matmul_lowprec import quant_matmul_248
from triteia.ao.ops.matmul.native_mm_lowprec import native_matmul_lowprec_248

marlin_tensors_file = ".local/tinyllama.tinyllama-1.1b-chat-v1.0/marlin.safetensors"
gptq_tensors_file   = ".local/tinyllama.tinyllama-1.1b-chat-v1.0/gptq.safetensors"
marlin_tensors = {}
gptq_tensors = {}
device = "cuda:0"
prefix = "model.layers.0.self_attn.q_proj"

with st.safe_open(
    marlin_tensors_file, framework="pt", device=device
) as f:
    for key in f.keys():
        if key.startswith(prefix):
            module_name = key.removeprefix(prefix + ".")
            marlin_tensors[module_name] = f.get_tensor(key)

with st.safe_open(
    gptq_tensors_file, framework="pt", device=device
) as f:
    for key in f.keys():
        if key.startswith(prefix):
            module_name = key.removeprefix(prefix + ".")
            gptq_tensors[module_name] = f.get_tensor(key)

print(marlin_tensors.keys())
print(gptq_tensors.keys())
# ---
print(gptq_tensors['g_idx'].shape)
input_dim = gptq_tensors['g_idx'].shape[0]

x = torch.rand((1, input_dim), dtype=torch.float16, device=device)
triton_output = quant_matmul_248(
    4, x, 
    gptq_tensors['qweight'],
    gptq_tensors['qzeros'],
    gptq_tensors['scales'],
    gptq_tensors['g_idx'],
    bias=None
)
torch_output = native_matmul_lowprec_248(
    4, x, 
    gptq_tensors['qweight'],
    gptq_tensors['qzeros'],
    gptq_tensors['scales'],
    gptq_tensors['g_idx'],
    bias=None
)
print(gptq_tensors['qweight'].shape)
print(triton_output)
print(torch_output)
# -- marlin computes
workspace = torch.zeros(input_dim//128*16, device=device, dtype=torch.int)
output = torch.zeros((1, input_dim), device=device, dtype=torch.float16)

marlin_layer = marlin.Layer(
    infeatures = input_dim,
    outfeatures = input_dim,
    groupsize=-1,
)
marlin_layer.B = marlin_tensors['B'].to(device)
marlin_layer.s = marlin_tensors['s'].to(device)
marlin_layer.workspace = workspace
output = marlin_layer(x)
print(output)