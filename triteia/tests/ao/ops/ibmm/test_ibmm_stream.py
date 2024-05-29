import torch
import safetensors as st
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin, ibmm_sparse_marlin_stream
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.ao.utils.distribution import generate_model_distribution

prefix = "model.layers.0.mlp.down_proj"
gptq_tensors_file = ".local/tinyllama/model.safetensors"
gptq_tensors = {}

DEV = "cuda:0"

with st.safe_open(
    gptq_tensors_file, framework="pt", device=DEV
) as f:
    for key in f.keys():
        if key.startswith(prefix):
            module_name = key.removeprefix(prefix + ".")
            gptq_tensors[module_name] = f.get_tensor(key)
dequantized_weight = dequantize_weight(
    gptq_tensors["qweight"],
    gptq_tensors["qzeros"],
    gptq_tensors["scales"],
).to(torch.float16)
scales = gptq_tensors["scales"]

if __name__=="__main__":
    k = 2048
    m = 4096
    num_requests = 25
    num_models = 10
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
    print(ref_output)
    stream_output = torch.zeros((num_requests, m), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin_stream(
        4, indices, metas, stream_output, x, qs, scales
    )
    print(stream_output)