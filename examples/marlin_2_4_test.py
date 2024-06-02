import torch
import torch.nn as nn
import safetensors as st
from triteia.lib.marlin import Layer_2_4
from triteia.lib.marlin.utils import quant_4_nt
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.ao.ops.matmul.native_mm_lowprec import native_matmul_lowprec_248

prefix = "model.layers.0.mlp.down_proj"
gptq_tensors_file = ".local/tinyllama/gptq.safetensors"
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

if __name__=="__main__":
    k = 5632
    m = 2048
    n = 1
    groupsize = -1
    workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)

    A = torch.randn((n, k), dtype=torch.half, device=DEV)
    fp16_weight = dequantized_weight
    s = scales

    print(f"fp16_weight: {fp16_weight.shape}, s: {s.shape}")
    print(f"fp16_weight: {fp16_weight}, s: {s}")

    B, s, meta = quant_4_nt(fp16_weight, s, groupsize=groupsize)
    C_ref = torch.matmul(A, fp16_weight.t())

    ref_output = native_matmul_lowprec_248(4, A, gptq_tensors['qweight'], gptq_tensors['qzeros'], gptq_tensors['scales'], gptq_tensors['g_idx'], bias=None)

    layer = Layer_2_4(k, m, groupsize =-1)

    layer.B = B
    layer.meta = meta
    layer.s = s
    layer.workspace = workspace
    print(f"layer.B: {layer.B.shape}, layer.s: {layer.s.shape}, layer.meta: {layer.meta.shape}")
    C = layer(A)

    torch.cuda.synchronize()
    print(f"C: {C}")
    print(f"ref_output: {ref_output}")
    print(f"C_ref: {C_ref}")
    print(f"max diff: {(C-ref_output).max()}")