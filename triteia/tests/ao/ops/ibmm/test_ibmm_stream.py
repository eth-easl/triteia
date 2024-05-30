import torch
import safetensors as st
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.ao.utils.distribution import generate_model_distribution

DEV="cuda:0"

if __name__=="__main__":
    k = 4096
    m = 5504
    num_requests = 5
    num_models = 1
    distribution = "uniform"
    indices = generate_model_distribution(distribution, num_requests, num_models)
    indices = torch.sort(indices)[0]
    print(f"indices: {indices}")
    fp16, qs, scales, metas = generate_2_4_pruned(num_models, m, k)
    groupsize = -1
    workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)
    x = torch.randn((num_requests, k), dtype=torch.float16, device=DEV)
    ref_output = torch.zeros((num_requests, m), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin(
        4,indices, metas, ref_output, x, qs, scales
    )
    stream_output = torch.zeros((num_requests, m), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin_stream(
        4, indices, metas, stream_output, x, qs, scales, parallel=False
    )
    for i in range(num_requests):
        if not torch.allclose(ref_output[i], stream_output[i]):
            print(f"Error starts at row {i}")
            break