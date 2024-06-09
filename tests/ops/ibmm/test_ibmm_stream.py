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
<<<<<<< HEAD
    k = 5504 # in_feature
    m = 4096 # outfeature
    num_requests = 5
    num_models = 4
=======
    k = 2048 # in_feature
    m = 2048 # outfeature
    num_requests = 8
    num_models = 6
>>>>>>> bce058b381fb754e222c17684c5b3e8d24357434
    distribution = "uniform"
    indices = generate_model_distribution(distribution, num_requests, num_models)
    indices = torch.sort(indices)[0]
    # indices = torch.tensor([0] * 16, device=DEV, dtype=torch.int32)
    # indices = torch.cat((indices, torch.tensor([1] * 16, device=DEV, dtype=torch.int32)))
    # indices = torch.tensor([0] * 16, device=DEV, dtype=torch.int32)
<<<<<<< HEAD
    indices = torch.tensor([0,1,2,3,3], device=DEV, dtype=torch.int32)
    
=======
    indices = torch.tensor([-1,-1,3,5,2,4,0,1], device=DEV, dtype=torch.int32)
>>>>>>> bce058b381fb754e222c17684c5b3e8d24357434
    print(f"indices: {indices}")
    fp16, qs, scales, metas = generate_2_4_pruned(num_models, m, k)
    groupsize = -1
    print(f"qs: {qs[0][0][0:10]}")
    # print(f"qs: {qs[1][0][0:10]}")
    
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
    if len(wrong_rows) == 0:
        print("All rows are correct")
        print(f"ref_output: {ref_output}")
        print(f"stream_output: {stream_output}")
    print(f"Wrong rows: {wrong_rows}")