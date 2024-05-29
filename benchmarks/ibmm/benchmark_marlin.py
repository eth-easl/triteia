import torch
from timeit import default_timer as timer
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.ops.ibmm.ibmm_fp16 import ibmm_fp16

def benchmark(K, M, num_reqs, num_models, dist):
    DEV="cuda:0"
    
    fp16, qs, scales, metas = generate_2_4_pruned(
        num_models,
        M, K, groupsize=-1, device=DEV
    )
    x = torch.randn((num_requests, K), dtype=torch.float16, device=DEV)
    # warmup here
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    torch.cuda.synchronize()

if __name__ == "__main__":
    Ks = [2048, 4096]
    Ms = [2048, 4096]
    num_requests = [100]
    num_models = [10]
    distribution = ['uniform']
    for K in Ks:
        for M in Ms:
            for num_req in num_requests:
                for num_model in num_models:
                    for dist in distribution:
                        benchmark(K, M, num_req, num_model, dist)