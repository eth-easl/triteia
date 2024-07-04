import torch
from triteia.python.ops import (
    bmm_4bit_2_4,
    bmm_4bit_2_4_forloop,
    gen_batched_sparse_quant4_NT
)

dev = "cuda"
b=16
n=16
m=35328
p=6656
groupsize = -1

x = torch.randn((b,1,  p), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
    b, m, p, groupsize=groupsize, device=dev
)
fp16_output = torch.bmm(x, weight_ref)
forloop_output = bmm_4bit_2_4_forloop(qweight, x, meta, scale)
native_output = bmm_4bit_2_4(qweight, x, meta, scale)
print(fp16_output)
print(forloop_output)
print(native_output)
print(f"native_output: {native_output.shape}, fp16_output: {fp16_output.shape}, forloop_output: {forloop_output.shape}")
torch.cuda.synchronize()
