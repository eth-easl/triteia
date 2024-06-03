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
    base_weight = fp16[0]
    x = torch.randn((num_reqs, K), dtype=torch.float16, device=DEV)
    
    indices = generate_model_distribution(dist, num_reqs, num_models)
    indices = torch.sort(indices)[0]
    indices = torch.tensor([-1,-1, 3, 1]).to(DEV)
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
    output = ibmm_fp16(indices, None, fp16_output, x, fp16, None)
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    fp16_time = start.elapsed_time(end)
    
    # baseline2: for loop
    # warmup here
    ibmm_sparse_marlin(
        4, indices, metas, None, x, qs, scales, base_weight=base_weight
    )
    # actual measure
    torch.cuda.nvtx.range_push("ibmm_sparse_marlin naive")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    ref_output = ibmm_sparse_marlin(
        4, indices, metas, None, x, qs, scales, base_weight=base_weight
    )
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    for_loop_time = start.elapsed_time(end)
    
    # sparse Marlin
    # warmup here
    ibmm_sparse_marlin_stream(
        4,indices, metas, None, x, qs, scales, base_weight=base_weight, parallel=False
    )
    # actual measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push("ibmm_sparse_marlin_stream parallel=False")
    start.record()
    output = ibmm_sparse_marlin_stream(
        4,indices, metas, None, x, qs, scales, base_weight=base_weight, parallel=False
    )
    end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    stream_time = start.elapsed_time(end)
    
    # # sparse_marlin parallel
    # # warmup here
    # parallel_stream_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    # parallel_stream_output = ibmm_sparse_marlin_stream(
    #     4,indices, metas, parallel_stream_output, x, qs, scales, parallel=True
    # )
    # # actual measure
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # parallel_stream_output = torch.zeros((num_reqs, M), dtype=torch.float16, device=DEV)
    # torch.cuda.nvtx.range_push("ibmm_sparse_marlin_stream parallel=True")
    # start.record()
    # parallel_stream_output = ibmm_sparse_marlin_stream(
    #     4,indices, metas, parallel_stream_output, x, qs, scales, parallel=True
    # )
    # end.record()
    # torch.cuda.synchronize()
    # torch.cuda.nvtx.range_pop()
    # parallel_stream_time = start.elapsed_time(end)
    result.append({
        "M": M,
        "K": K,
        "num_reqs": num_reqs,
        "num_models": num_models,
        "dist": dist,
        "func_for_loop": for_loop_time,
        "func_improved": stream_time,
        "func_fp16": fp16_time,
        #  "func_parallel": parallel_stream_time,
    })
    
    ## verify resutlts...
    # if not torch.allclose(ref_output, parallel_stream_output):
    #     print("error: ref_output != parallel_stream_output")
    #     pass
    if not torch.allclose(ref_output, output):
        print("error: ref_output != output")
        print(ref_output)
        print(output)
    return result
    
if __name__ == "__main__":
    import pandas as pd
    Ks = [4096]
    Ms = [4096]
    num_requests = [4]
    num_models = [4]
    distribution = ['uniform']
    trials = 1
    results = []
    for K in Ks:
        for M in Ms:
            for num_req in num_requests:
                for num_model in num_models:
                    for dist in distribution:
                        for i in range(trials):
                            res = benchmark(K, M, num_req, num_model, dist)
                            results.extend(res)
    results = pd.DataFrame(results)
    print(results)
    # results.to_csv(".local/benchmark_marlin.csv", index=False)