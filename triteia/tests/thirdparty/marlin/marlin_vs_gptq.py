import os
import torch
import safetensors as st
import triteia.lib.marlin as marlin
from triteia.ao.ops.matmul.matmul_lowprec import quant_matmul_248
from triteia.ao.ops.matmul.native_mm_lowprec import native_matmul_lowprec_248

marlin_tensors_file = ".local/tinyllama/marlin.safetensors"
gptq_tensors_file   = ".local/tinyllama/model.safetensors"
marlin_tensors = {}
gptq_tensors = {}
device = "cuda:0"
prefix = "model.layers.1.self_attn.q_proj"

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
# ---
input_dim = gptq_tensors['g_idx'].shape[0]
print(f"input_dim: {input_dim}")
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
# print(triton_output)
# print(torch_output)

# -- marlin computes
workspace = torch.zeros(input_dim//128*16, device=device, dtype=torch.int)
marlin_layer = marlin.Layer_2_4(
    infeatures = input_dim,
    outfeatures = input_dim,
    groupsize=-1,
)
marlin_layer.B = marlin_tensors['qweight'].to(device)
marlin_layer.s = marlin_tensors['scales'].to(device)
marlin_layer.meta = marlin_tensors['meta'].to(device)

print(f"marlin_layer.B: {marlin_layer.B.shape}, marlin_layer.s: {marlin_layer.s.shape}, marlin_layer.meta: {marlin_layer.meta.shape}")

marlin_layer.workspace = workspace
C = torch.zeros((1, input_dim), dtype=torch.half, device=device)
# output = marlin.mul_2_4(
#     x.view((-1, x.shape[-1])),
#     marlin_layer.B,
#     marlin_layer.meta,
#     C,
#     marlin_layer.s,
#     workspace,
# )
output = marlin_layer(x)
print(f"max diff triton-torch: {(triton_output - torch_output).max()}")
print(f"max diff: {(output - triton_output).max()}")