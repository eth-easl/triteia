import torch
import torch.nn as nn
import safetensors as st
from triteia.lib.marlin import Layer_2_4
from triteia.lib.marlin.utils import quant_4_nt
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.ao.ops.matmul.native_mm_lowprec import native_matmul_lowprec_248
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
    k = 5632
    m = 2048
    num_requests = 100
    num_models = 16
    distribution = "uniform"
    indices = generate_model_distribution(distribution, num_requests, num_models)
    indices = torch.sort(indices)[0]
    
    groupsize = -1
    workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)