import torch
from timeit import default_timer as timer
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_marlin, ibmm_marlin_vectorized, ibmm_marlin_native
from triteia.ao.utils.distribution import generate_model_distribution
from triteia.ao.utils.gen_tensor import gen_quant4, gen_pruned_quant4_NT

def prepare_quantized_weights(bitwidth, K, N, num_models):
    refs = []
    qweights = []
    scales = []
    for i in range(num_models):
        ref, qweight, scale = gen_quant4(K, N)
        refs.append(ref)
        qweights.append(qweight)
        scales.append(scale)
    refs = torch.stack(refs)
    qweight = torch.stack(qweights)
    scales = torch.stack(scales)
    return refs, qweight, scales

def prepare_fp16_weights(K, N, num_models):
    weights = []
    for i in range(num_models):
        weight = torch.rand((K, N), dtype=torch.float16, device="cuda")
        weights.append(weight)
    return torch.stack(weights)

def _bench_ibmm(bitwidth, indices, y, x, qweight, scales, base_weight):
    torch.cuda.synchronize()
    start = timer()
    y = torch.matmul(x, base_weight.T)
    ibmm_marlin(bitwidth, indices, y, x, qweight, scales)
    torch.cuda.synchronize()
    end = timer()
    return (end-start) * 1000

def _bench_ibmm_vectorized(bitwidth, indices, y, x, qweight, scales, base_weight):
    torch.cuda.synchronize()
    start = timer()
    y = torch.matmul(x, base_weight.T)
    ibmm_marlin_vectorized(bitwidth, indices, y, x, qweight, scales)
    torch.cuda.synchronize()
    end = timer()
    return (end-start) * 1000

def _bench_fp16(indices, y, x, weight):
    torch.cuda.synchronize()
    start_timer = timer()
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices, counts = torch.unique(valid_indices, sorted=False, return_counts=True)
    start = 0
    for id, count in zip(unique_indices, counts):
        idx_mask = indices == id
        inp = x[idx_mask]
        y[start:start+count, :] += torch.matmul(inp, weight[id].T)
        start += count
    torch.cuda.synchronize()
    end_timer = timer()
    return (end_timer-start_timer) * 1000

def benchmark():
    Ns = [512]
    Ks = [512]
    bitwidth = 4
    max_num_models = [16]
    num_reqs = 100
    distribution = "uniform"
    
    for num_models in max_num_models:
        for N in Ns:
            for K in Ks:
                # generate num_models quantized weights
                fp16_weights, qweight, scales = prepare_quantized_weights(bitwidth, K, N, num_models)
                x = torch.rand((num_reqs, K), dtype=torch.float16, device="cuda")
                indices = generate_model_distribution(distribution, num_reqs, num_models)
                # sort indices
                
                indices = torch.sort(indices)[0]
                print(indices)
                y_1 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
                y_2 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
                torch.cuda.synchronize()
                
                # warmup
                ibmm_marlin(bitwidth, indices, y_1, x, qweight, scales)
                _bench_ibmm_vectorized(bitwidth, indices, y_1, x, qweight, scales, fp16_weights[0])
                _bench_fp16(indices, y_2, x, fp16_weights)
                torch.cuda.synchronize()
                print("Warmup Done")
                # nvtx
                torch.cuda.nvtx.range_push("bench ibmm")
                v1_elapsed = _bench_ibmm(bitwidth, indices, y_1, x, qweight, scales, fp16_weights[0])
                torch.cuda.nvtx.range_pop()
                
                torch.cuda.nvtx.range_push("bench fp16")
                fp16_elapsed = _bench_fp16(indices, y_2, x, fp16_weights)
                torch.cuda.nvtx.range_pop()
                
                torch.cuda.nvtx.range_push("bench ibmm vectorized")
                v2_elapsed = _bench_ibmm_vectorized(bitwidth, indices, y_1, x, qweight, scales, fp16_weights[0])
                torch.cuda.nvtx.range_pop()
                
                print("--"* 20)
                print(f"# models: {num_models}, # requests: {num_reqs}, N: {N}, K: {K}")
                print(f"v1: {v1_elapsed:.2f} ms")
                print(f"v2: {v2_elapsed:.2f} ms")
                print(f"fp16: {fp16_elapsed:.2f} ms")
                print("--"* 20)

if __name__=="__main__":
    benchmark()