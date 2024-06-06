import torch
import safetensors as st
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream, ibmm_native
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.ao.utils.distribution import generate_model_distribution

DEV="cuda:0"

if __name__=="__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=4)
    k = 4096 # in_feature
    m = 4096 # outfeature
    num_requests = 32
    num_models = 8
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
    stream_output = ibmm_native(
        4, indices, metas, stream_output, x, qs, scales
    )
    for i in range(num_requests):
        if not torch.allclose(ref_output[i], stream_output[i]):
            print(f"Error starts at row {i}")
            print(f"ref_output: {ref_output}")
            print(f"stream_output: {stream_output}")
            break