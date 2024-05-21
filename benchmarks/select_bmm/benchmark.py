import torch
from triteia.ao.utils.gen_tensor import generate_bitblas_weight
from triteia.ao.ops.linalg.select_matmul.select_bmm import ibmm
from triteia.ao.ops.linalg.select_matmul.select_bmm import vectorized_ibmm
from triteia.ao.utils.distribution import generate_model_distribution
from triteia.ao.ops.linalg.matmul.group_gemm_lowprec import group_gemm
from timeit import default_timer as timer
import bitblas

bitblas.set_log_level("DEBUG")

def _bench_ibmm(bitwidth, indices, y, x, qweight, qzero, scales):
    torch.cuda.synchronize()
    start = timer()
    ibmm(bitwidth, indices, y, x, qweight, qzero, scales)
    torch.cuda.synchronize()
    end = timer()
    return (end-start) * 1000

def _bench_fp16(bitwidth, indices, y, x, weight):
    start = timer()
    unique_indices = torch.unique(indices)
    for id in unique_indices:
        pass
    
def benchmark():
    Ns = [4096]
    Ks = [4096]
    bitwidth = 4
    max_num_models = [4]
    num_reqs = 100
    distribution = "uniform"
    
    for num_models in max_num_models:
        for N in Ns:
            for K in Ks:
                # generate num_models quantized weights
                qweights = []
                scales = []
                qzeros = []
                for i in range(num_models):
                    qweight, scale, qzero = generate_bitblas_weight(bitwidth, K, N, N)
                    qweights.append(qweight)
                    scales.append(scale)
                    qzeros.append(qzero)
                qweight = torch.stack(qweights)
                scales = torch.stack(scales)
                qzero = torch.stack(qzeros)
                print(f"qweight.shape: {qweight.shape}, scales.shape: {scales.shape}, qzero.shape: {qzero.shape}")
                x = torch.rand((num_reqs, K), dtype=torch.float16, device="cuda")
                indices = generate_model_distribution(distribution, num_reqs, num_models)
                print(indices)
                y_1 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
                y_2 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
                torch.cuda.synchronize()
                # warmup
                ibmm(bitwidth, indices, y_1, x, qweight, qzero, scales)
                group_gemm(bitwidth, indices, y_2, x, qweight, qzero, scales)
                torch.cuda.synchronize()
                print("Warmup Done")
                ## 
                v1_elapsed = _bench_ibmm(bitwidth, indices, y_1, x, qweight, qzero, scales)
                
                torch.cuda.synchronize()
                start = timer()
                group_gemm(bitwidth, indices, y_2, x, qweight, qzero, scales)
                torch.cuda.synchronize()
                end = timer()
                v2_elapsed = (end-start) * 1000
                
                print(f"# models: {num_models}, # requests: {num_reqs}")
                print(f"N={N}, K={K}")
                print(f"v1: {v1_elapsed:.2f} ms")
                print(f"v2: {v2_elapsed:.2f} ms")
                print("--"* 20)
               # assert torch.allclose(y_1, y_2)

if __name__=="__main__":
    benchmark()