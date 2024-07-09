import torch
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_sparse_quant4_NT
from triteia.python.ops.matmul.sbmm import sbmm_4bit_2_4_forloop

m = 4096
k = 4096
nr = 25
nm = 10
groupsize = -1
distribution = 'uniform'
dev = "cuda"


indices = generate_model_distribution(distribution, nr, nm)
indices = torch.sort(indices)[0]
x = torch.randn((nr, k), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
    nr, m, k, groupsize=groupsize, device=dev
)
sbmm_4bit_2_4_forloop(qweight, x, meta, scale, indices, base_weight=None)