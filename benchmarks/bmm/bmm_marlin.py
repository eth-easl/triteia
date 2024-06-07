import torch
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.ops.bmm import bmm_sparse_marlin_forloop
from triteia.ao.ops.bmm import bmm_native

DEV = "cuda:0"
torch.manual_seed(0)
def benchmark(K, M, num_reqs):
    fp16, qs, scales, metas = generate_2_4_pruned(
        num_reqs,
        M, K, groupsize=-1, device=DEV
    )
    x = torch.randn((num_reqs, K), dtype=torch.float16, device=DEV)
    ref_output = bmm_sparse_marlin_forloop(4, x, qs, scales, metas)
    
    native_output = bmm_native(4, x, qs, scales, metas)
    for i in range(num_reqs):
        if not torch.allclose(ref_output[i], native_output[i]):
            print(f"Error at row {i}")
            print(f"ref_output: {ref_output[i]}")
            print(f"native_output: {native_output[i]}")

    
if __name__=="__main__":
    num_reqs = 128
    M = 4096
    K = 4096
    benchmark(K, M, num_reqs)