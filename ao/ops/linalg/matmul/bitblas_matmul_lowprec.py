import torch
import bitblas
from fractions import Fraction
from ao.utils.dtypes import QUANTIZED_DTYPE
def bitblas_quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx, bias=None):
    pack_factor = Fraction(bitwidth, 32)
    infeatures = qweight.shape[0] // pack_factor
    outfeatures = qweight.shape[1]
    print(f"qweight.shape: {qweight.shape}, x.shape: {x.shape}")
    M = x.shape[0]
    N = x.shape[1]
    K = qweight.shape[1]
    matmul_config = bitblas.MatmulConfig(
        M = M,
        N = N,
        K = K,
        A_dtype = 'float16',
        W_dtype=QUANTIZED_DTYPE[bitwidth],
        accum_dtype='float16',
        out_dtype='float16',
        with_bias=False,
        group_size=infeatures,
        with_scaling=True,
        with_zeros=True,
        zeros_mode="quantized"
    )
    matmul = bitblas.Matmul(config=matmul_config)
    output_tensor = matmul(x, qweight, scale=scale, zeros=qzero)
    return output_tensor