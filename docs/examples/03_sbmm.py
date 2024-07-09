import torch
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_sparse_quant4_NT
from triteia.python.ops.matmul.sbmm import (
    sbmm_4bit_2_4_forloop,
    sbmm_4bit_2_4_native,
    sbmm_4bit_2_4_multilaunch,
    sbmm_16bit_forloop,
)

m = 12800
k = 4096
nr = 50
nm = 10
groupsize = -1
distribution = "uniform"
dev = "cuda"

indices = generate_model_distribution(distribution, nr, nm)
indices = torch.sort(indices)[0]
print(indices)
x = torch.randn((nr, k), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
    nr, m, k, groupsize=groupsize, device=dev
)
print(weight_ref.shape)
fp16_output = sbmm_16bit_forloop(weight_ref, x, indices, base_weight=None)
forloop_output = sbmm_4bit_2_4_forloop(
    qweight, x, meta, scale, indices, base_weight=None
)
native_output = sbmm_4bit_2_4_native(qweight, x, meta, scale, indices, base_weight=None)
multilaunch_output = sbmm_4bit_2_4_multilaunch(
    qweight, x, meta, scale, indices, base_weight=None
)
print(fp16_output)
print(forloop_output)
print(native_output)
print(multilaunch_output)

print("--" * 20)
print(
    f"native_output: {native_output.shape}, fp16_output: {fp16_output.shape}, forloop_output: {forloop_output.shape}, multilaunch_output: {multilaunch_output.shape}"
)
