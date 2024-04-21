import torch
import bitblas
from fractions import Fraction
from ao.utils.dtypes import QUANTIZED_DTYPE

def bitblas_quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx=None, bias=None):
    pack_factor = Fraction(bitwidth, 32)
    print(f"qweight.shape: {qweight.shape}, x.shape: {x.shape}")
    M = x.shape[0]
    N = x.shape[1]
    K = qweight.shape[1]
    matmul_config = bitblas.MatmulConfig(
        M = M,
        N = N,
        K = K,
        A_dtype = 'float16',
        W_dtype="uint4",
        accum_dtype='float16',
        out_dtype='float16',
        with_bias=False,
        group_size=qweight.shape[0],
        with_scaling=True,
        with_zeros=True,
        zeros_mode="quantized"
    )
    bitblas_linear = bitblas.Linear(
        in_features=qweight.shape[0],
        out_features=qweight.shape[1],
        bias=False,
        A_dtype="float16",  # activation A dtype
        W_dtype="uint4",  # weight W dtype
        accum_dtype="float16",  # accumulation dtype
        out_dtype="float16",  # output dtype
        # configs for weight only quantization
        group_size=qweight.shape[0],  # setting for grouped quantization
        with_scaling=True,  # setting for scaling factor
        with_zeros=True,  # setting for zeros
        zeros_mode="quantized",  # setting for how to calculating zeros
    )
    bitblas_linear.qweight = qweight
    bitblas_linear.zeros = qzero
    bitblas_linear.scales = scale
    # matmul = bitblas.Matmul(config=matmul_config)
    output_tensor = bitblas_linear(x)
    return output_tensor