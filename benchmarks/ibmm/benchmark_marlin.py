import torch
from triteia.ao.utils.distribution import generate_model_distribution
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.ops.ibmm.ibmm_fp16 import ibmm_fp16

DEV="cuda:0"

def benchmark(K, M, num_reqs, num_models, dist):
    result = []
    fp16, qs, scales, metas = generate_2_4_pruned(
        num_models,
        M, K, groupsize=-1, device=DEV
    )
    x = torch.randn((num_reqs, K), dtype=torch.float16, device=DEV)
    
    indices = generate_model_distribution(dist, num_reqs, num_models)
    indices = torch.sort(indices)[0]
    # baseline1: fp16: 
    # warmup here
    fp16_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    ibmm_fp16(indices, None, fp16_output, x, fp16, None)
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fp16_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    torch.cuda.nvtx.range_push("ibmm fp16")
    start.record()
    ibmm_fp16(indices, None, fp16_output, x, fp16, None)
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    fp16_time = start.elapsed_time(end)
    
    
    
    # baseline2: for loop
    # warmup here
    ref_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin(
        4, indices, metas, ref_output, x, qs, scales
    )
    ref_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    # actual measure
    torch.cuda.nvtx.range_push("ibmm_sparse_marlin naive")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    ibmm_sparse_marlin(
        4, indices, metas, ref_output, x, qs, scales
    )
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    for_loop_time = start.elapsed_time(end)
    
    # sparse Marlin
    # warmup here
    output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin_stream(
        4,indices, metas, output, x, qs, scales, parallel=False
    )
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    torch.cuda.nvtx.range_push("ibmm_sparse_marlin_stream parallel=False")
    start.record()
    output = ibmm_sparse_marlin_stream(
        4,indices, metas, output, x, qs, scales, parallel=False
    )
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    stream_time = start.elapsed_time(end)
    
    
    
    # sparse_marlin parallel
    # warmup here
    parallel_stream_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    parallel_stream_output = ibmm_sparse_marlin_stream(
        4,indices, metas, parallel_stream_output, x, qs, scales, parallel=True
    )
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    parallel_stream_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    torch.cuda.nvtx.range_push("ibmm_sparse_marlin_stream parallel=True")
    start.record()
    parallel_stream_output = ibmm_sparse_marlin_stream(
        4,indices, metas, parallel_stream_output, x, qs, scales, parallel=True
    )
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    parallel_stream_time = start.elapsed_time(end)
    result.append({
        "M": M,
        "K": K,
        "num_reqs": num_reqs,
        "num_models": num_models,
        "dist": dist,
        "for_loop_time": for_loop_time,
        "improved": stream_time,
        "fp16_time": fp16_time,
        "parallel_time": parallel_stream_time,
    })
    
    ## verify resutlts...
    if not torch.allclose(ref_output, parallel_stream_output):
        print("error: ref_output != parallel_stream_output")
        # print(f"ref: {ref_output}")
        # print(f"output: {output}")
        # raise RuntimeError(f"Error at M={M}, K={K}, num_reqs={num_reqs}, num_models={num_models}, dist={dist}")
        pass
    if not torch.allclose(ref_output, output):
        print("error: ref_output != output")
        # print(f"ref: {ref_output}")
        # print(f"output: {output}")
        # raise RuntimeError(f"Error at M={M}, K={K}, num_reqs={num_reqs}, num_models={num_models}, dist={dist}")
        pass
    return result
    
if __name__ == "__main__":
    import pandas as pd
    Ks = [4096]
    Ms = [4096]
    num_requests = [100]
    num_models = [2,4,6,8,16]
    distribution = ['uniform','zipf:1.5']
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