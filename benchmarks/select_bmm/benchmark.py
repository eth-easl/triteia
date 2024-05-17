import torch
from triteia.ao.utils.gen_tensor import generate_bitblas_weight
from triteia.ao.ops.linalg.select_matmul.select_bmm import ibmm
from triteia.ao.ops.linalg.select_matmul.select_bmm import vectorized_ibmm

from timeit import default_timer as timer
def benchmark():
    Ns = [4096]
    Ks = [4096]
    bitwidth = 4
    num_models = 4
    num_reqs = 16
    
    for N in Ns:
        for K in Ks:
            qweight, scales, qzero = generate_bitblas_weight(bitwidth, K, N, N)
            qweight = qweight.repeat(num_models, 1, 1)
            scales = scales.repeat(num_models, 1,1)
            qzero = qzero.repeat(num_models, 1,1)
            print(f"qweight.shape: {qweight.shape}, scales.shape: {scales.shape}, qzero.shape: {qzero.shape}")
            x = torch.rand((num_reqs, K), dtype=torch.float16, device="cuda")
            
            indices = torch.randint(-1, num_models, (num_reqs,), device="cuda")
            print(indices)
            y_1 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
            start = timer()
            ibmm(bitwidth, indices, y_1, x, qweight, qzero, scales)
            end = timer()
            print(f"Time taken for N={N}, K={K}: {end-start} seconds")
            y_2 = torch.zeros((num_reqs, N), dtype=torch.float16, device="cuda")
            start = timer()
            vectorized_ibmm(bitwidth, indices, y_2, x, qweight, qzero, scales)
            end = timer()
            print(f"Time taken for N={N}, K={K}: {end-start} seconds")
            assert torch.allclose(y_1, y_2)
            print(y_1)
            print(y_2)
            
if __name__=="__main__":
    benchmark()