import torch
import numpy as np
import safetensors as st
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream, ibmm_native
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.ao.utils.distribution import generate_model_distribution

DEV="cuda:0"

if __name__=="__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=4)
    k = 256 # in_feature
    m = 256 # outfeature
    num_requests = 64
    num_models = 2
    distribution = "zipf:2"
    indices = generate_model_distribution(distribution, num_requests, num_models)
    indices = torch.sort(indices)[0]
    
    fp16, qs, scales, metas = generate_2_4_pruned(num_models, m, k)
    groupsize = -1
    
    x = torch.randn((num_requests, k), dtype=torch.float16, device=DEV)
    ref_output = torch.zeros((num_requests, m), dtype=torch.float16, device=DEV)
    ref_output = ibmm_sparse_marlin( 
        4, indices, metas, ref_output, x, qs, scales
    )
    stream_output = torch.zeros((num_requests, m), dtype=torch.float16, device=DEV)
    stream_output = ibmm_native(
        4, indices, metas, stream_output, x, qs, scales
    )
    wrong_rows = []
    for i in range(num_requests):
        if not torch.allclose(ref_output[i], stream_output[i]):
            print(f"Error at row {i}, indices={indices[i]}")
            wrong_rows.append(i)
            print(f"ref_output: {ref_output[i]}")
            print(f"stream_output: {stream_output[i]}")
    print(f"Wrong rows: {wrong_rows}")