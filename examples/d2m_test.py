import torch
import safetensors as st
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.lib.marlin import Layer_2_4, mask_creator

prefix = "model.layers.0.mlp.down_proj"
gptq_tensors_file = ".local/tinyllama/model.safetensors"

marlin_tensors = {}
gptq_tensors = {}
DEV = "cuda:0"

def repack(dequantized_weight, scales):
    m = dequantized_weight.shape[1]
    k = dequantized_weight.shape[0]
    k_sp = k // 2
    group_size = k
    layer = Layer_2_4(
        dequantized_weight.shape[1],
        dequantized_weight.shape[0],
        groupsize= -1,
    )
    group_size = k
    layer.n = m
    layer.k = k
    layer.groupsize = group_size
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=DEV)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
    layer.s = torch.empty((k_sp // (group_size // 2), m), dtype=torch.half, device=DEV)
    layer.pack(dequantized_weight, scales, True)
    return layer.B, layer.s, layer.meta
    
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
).to(torch.float16).t()
# print sparsity of dequantized weight
print(f"Sparsity: {dequantized_weight.nonzero().shape[0] / dequantized_weight.numel()}")
scales = gptq_tensors["scales"]

print(f"weight shape: {dequantized_weight.shape}, scales shape: {scales.shape}")

input_dim = dequantized_weight.shape[1]

k = dequantized_weight.shape[0]
m = dequantized_weight.shape[1]
n = 1

x = torch.rand((n, input_dim), dtype=torch.float16, device=DEV)

B, s, meta = repack(dequantized_weight, scales)
ref_output = torch.matmul(x, dequantized_weight)

layer = Layer_2_4(k, m, groupsize =-1)
workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)

layer.B = B
layer.meta = meta
layer.s = s
layer.workspace = workspace
C = layer(x)

print(f"Diff: {(C-ref_output).mean()}")
