import torch
from triteia.ao.utils.distribution import generate_model_distribution
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.ops.ibmm.ibmm_fp16 import ibmm_fp16

def benchmark(K, M, num_reqs, num_models, dist):
    DEV="cuda:0"
    result = []
    fp16, qs, scales, metas = generate_2_4_pruned(
        num_models,
        M, K, groupsize=-1, device=DEV
    )
    x = torch.randn((num_reqs, K), dtype=torch.float16, device=DEV)
    indices = generate_model_distribution(dist, num_reqs, num_models)
    indices = torch.sort(indices)[0]
    # warmup here
    ref_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin(
        4,indices, metas, ref_output, x, qs, scales
    )
    ref_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    ibmm_sparse_marlin(
        4,indices, metas, ref_output, x, qs, scales
    )
    end.record()
    torch.cuda.synchronize()
    for_loop_time = start.elapsed_time(end)
    
    # warmup here
    output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin_stream(
        4,indices, metas, output, x, qs, scales
    )
    
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    start.record()
    ibmm_sparse_marlin_stream(
        4,indices, metas, output, x, qs, scales
    )
    end.record()
    torch.cuda.synchronize()
    stream_time = start.elapsed_time(end)
    
    # warmup here
    fp16_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    ibmm_fp16(indices, None, fp16_output, x, fp16, None)
    
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fp16_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    start.record()
    ibmm_fp16(indices, None, fp16_output, x, fp16, None)
    end.record()
    torch.cuda.synchronize()
    fp16_time = start.elapsed_time(end)
    result.append({
        "M": M,
        "K": K,
        "num_reqs": num_reqs,
        "num_models": num_models,
        "dist": dist,
        "for_loop_time": for_loop_time,
        "stream_time": stream_time,
        "fp16_time": fp16_time,
    })
    if not torch.allclose(ref_output, output):
        raise RuntimeError("Error")
    return result
    
if __name__ == "__main__":
    import pandas as pd
    Ks = [4096]
    Ms = [4096]
    num_requests = [100]
    num_models = [2,4,8,16,32]
    distribution = ['uniform']
    results = []
    for K in Ks:
        for M in Ms:
            for num_req in num_requests:
                for num_model in num_models:
                    for dist in distribution:
                        res = benchmark(K, M, num_req, num_model, dist)
                        results.extend(res)
    results = pd.DataFrame(results)
    print(results)