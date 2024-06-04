import torch
from triteia.lib.marlin import bmm_2_4
from triteia.utils.generator import generate_2_4_pruned

DEV = "cuda:0"

def benchmark(K, M, num_reqs):
    fp16, qs, scales, metas = generate_2_4_pruned(
        num_reqs,
        M, K, groupsize=-1, device=DEV
    )
    x = torch.randn((num_reqs, K), dtype=torch.float16, device=DEV)



if __name__=="__main__":
    num_reqs = 100
    M = 1024
    K = 1024
    benchmark(K, M, num_reqs)