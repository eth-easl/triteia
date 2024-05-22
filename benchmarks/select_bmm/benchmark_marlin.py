import torch
import bitblas
from timeit import default_timer as timer
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_marlin, ibmm_marlin_native
from triteia.ao.utils.distribution import generate_model_distribution
from triteia.ao.utils.gen_tensor import gen_quant4, gen_pruned_quant4_NT

bitblas.set_log_level("DEBUG")

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
    # y = torch.matmul(x, base_weight.T)
    ibmm_marlin_native(indices, y, x, qweight, scales)
    torch.cuda.synchronize()
    end = timer()
    print(y)
    return (end-start) * 1000

def _bench_fp16(indices, y, x, weight):
    torch.cuda.synchronize()
    start = timer()
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices = torch.unique(valid_indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = torch.matmul(inp, weight[id].T)
        y[idx_mask] += output
    torch.cuda.synchronize()
    end = timer()
    return (end-start) * 1000

def benchmark():
    Ns = [4096]
    Ks = [4096]
    bitwidth = 4
    max_num_models = [20]
    num_reqs = 100
    distribution = "uniform"
    for num_models in max_num_models:
        for N in Ns:
            for K in Ks:
                # generate num_models quantized weights
                fp16_weights, qweight, scales = prepare_quantized_weights(bitwidth, K, N, num_models)
                
                x = torch.rand((num_reqs, K), dtype=torch.float16, device="cuda")

                indices = generate_model_distribution(distribution, num_reqs, num_models)
                print(indices)
                y_1 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
                y_2 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
                torch.cuda.synchronize()

                # warmup
                ibmm_marlin(bitwidth, indices, y_1, x, qweight, scales)
                _bench_fp16(indices, y_2, x, fp16_weights)
                torch.cuda.synchronize()
                print("Warmup Done")
                ## 
                v1_elapsed = _bench_ibmm(bitwidth, indices, y_1, x, qweight, scales, fp16_weights[0])
                fp16_elapsed = _bench_fp16(indices, y_2, x, fp16_weights)
                print("--"* 20)
                print(f"# models: {num_models}, # requests: {num_reqs}, N: {N}, K: {K}")
                print(f"v1: {v1_elapsed:.2f} ms")
                print(f"fp16: {fp16_elapsed:.2f} ms")
                print("--"* 20)

if __name__=="__main__":
    benchmark()