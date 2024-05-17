import os
import torch
import logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, TextGenerationPipeline
import safetensors as st
from safetensors.torch import save_file
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)
import bitblas
from bitblas import Matmul

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

bitblas.set_log_level("DEBUG")

bitwidth = 2
pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = f".local/opt-125m-{bitwidth}bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=bitwidth,  # quantize model to 2-bit
    group_size=128,  # it is recommended to set the value to 128,
    sym=False,
)

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

model.quantize(examples)
# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)

module_name = "model.decoder.layers.9.self_attn.q_proj"

with st.safe_open(os.path.join(quantized_model_dir, f"gptq_model-{bitwidth}bit-128g.safetensors"), "pt", device="cuda") as f:
    keys = f.keys()
    tensors = {key: f.get_tensor(key) for key in keys if module_name in key}

infeatures = 768
outfeatures = 768

cuda_old_linear = CudaOldQuantLinear(
    bits=bitwidth,
    group_size=128,
    infeatures=infeatures,
    outfeatures=outfeatures,
    bias=False,
)
cuda_old_linear.qweight = tensors[f"{module_name}.qweight"]
cuda_old_linear.qzeros = tensors[f"{module_name}.qzeros"]
cuda_old_linear.scales = tensors[f"{module_name}.scales"]
cuda_old_linear.g_idx = tensors[f"{module_name}.g_idx"]

bitblas_linear = bitblas.Linear(
    in_features=infeatures,
    out_features=outfeatures,
    bias=False,
    A_dtype="float16",
    W_dtype=f"uint{bitwidth}",
    accum_dtype="float16",
    out_dtype="float16",
    group_size=128,
    with_scaling=True,
    with_zeros=True,
    zeros_mode="quantized",
)
matmul_config = bitblas.MatmulConfig(
    M=1,
    N=outfeatures,
    K=infeatures,
    fast_decoding=True,
    A_dtype="float16",
    W_dtype=f"uint{bitwidth}",
    accum_dtype="float16",
    out_dtype="float16",
    layout="nt",
    with_bias=False,
    group_size=128,
    with_scaling=True,
    with_zeros=True,
    zeros_mode="quantized",
)
matmul = Matmul(matmul_config)
bitblas_linear.repack_from_gptq(cuda_old_linear)
print("repack done")

tensors = {
    "qweight": bitblas_linear.qweight,
    "scales": bitblas_linear.scales,
    "qzeros": bitblas_linear.zeros,
}

save_file(tensors, os.path.join(quantized_model_dir,"bitblas.safetensors"))
with st.safe_open(os.path.join(quantized_model_dir, "bitblas.safetensors"), "pt", device="cuda") as f:
    keys = f.keys()
    tensors = {key: f.get_tensor(key) for key in keys}
    
bitblas_linear.qweight = tensors["qweight"]
bitblas_linear.scales = tensors["scales"]
bitblas_linear.zeros = tensors["qzeros"]

print("BitBLAS quantized weight: ", bitblas_linear.qweight.shape)
print("BitBLAS quantized scales: ", bitblas_linear.scales.shape)
print("BitBLAS quantized zeros: ", bitblas_linear.zeros.shape)

inp = torch.rand(1, infeatures, dtype=torch.float16, device="cuda")

cuda_old_linear = cuda_old_linear.to("cuda")
res_cuda_old = cuda_old_linear(inp)
print(f"CudaOldQuantLinear output: {res_cuda_old}")

res_bitblas = matmul(
    inp, 
    bitblas_linear.qweight, 
    scale=bitblas_linear.scales,
    zeros=bitblas_linear.zeros
)

print(f"BitBLAS output: {res_bitblas}")
torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1, atol=1)