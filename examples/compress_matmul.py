import os
import torch
import logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, TextGenerationPipeline
import safetensors as st
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)
import bitblas
from bitblas import Matmul

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = ".local/opt-125m-2bit"

# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
# examples = [
#     tokenizer(
#         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
#     )
# ]

# quantize_config = BaseQuantizeConfig(
#     bits=2,  # quantize model to 4-bit
#     group_size=128,  # it is recommended to set the value to 128
#     desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
# )

# model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# model.quantize(examples)
# # save quantized model using safetensors
# model.save_quantized(quantized_model_dir, use_safetensors=True)

# # load quantized model to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

module_name = "model.decoder.layers.9.self_attn.q_proj"
with st.safe_open(os.path.join(quantized_model_dir, "gptq_model-2bit-128g.safetensors"), "pt") as f:
    keys = f.keys()
    tensors = {key: f.get_tensor(key) for key in keys if module_name in key}

infeatures = 48 * 16
outfeatures = 768

cuda_old_linear = CudaOldQuantLinear(
    bits=2,
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
    W_dtype="uint2",
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
    W_dtype="uint2",
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
print("repacking done")
inp = torch.rand(1, infeatures, dtype=torch.float16, device="cuda")

cuda_old_linear = cuda_old_linear.to("cuda")
res_cuda_old = cuda_old_linear(inp)
res_bitblas = matmul(
    inp,
    bitblas_linear.qweight,
    bitblas_linear.scales,
    bitblas_linear.zeros
)
# res_bitblas = bitblas_linear(inp)
print(f"CudaOldQuantLinear output: {res_cuda_old}")
print(f"BitBLAS output: {res_bitblas}")
# torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1, atol=1)