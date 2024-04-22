import torch
import bitblas
from fractions import Fraction
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE

global_matmul_registry = {}

def bitblas_quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx=None, bias=None):
    M = x.shape[0]
    N = x.shape[1]
    K = qweight.shape[0]

    assert qweight.shape[1]*2 == x.shape[1], f"2*qweight.shape[1] != x.shape[1], got {qweight.shape[1]*2} != {x.shape[1]}"
    assert qweight.shape[0] == qzero.shape[1]*2, f"qweight.shape[0] != qzero.shape[1], got {qweight.shape[0]} != {qzero.shape[1]}"
    assert qzero.shape[1] * 2 == scale.shape[0], f"qzero.shape[1] * 2 != scale.shape[0], got {qzero.shape[1] * 2} != {scale.shape[0]}"

    if f'{bitwidth},{M},{N},{K}' in global_matmul_registry:
        matmul = global_matmul_registry[f'{bitwidth},{M},{N},{K}']
    else:
        print(f"new signature: {bitwidth},{M},{N},{K}")
        matmul_config = bitblas.MatmulConfig(
            M=M,
            N=N,
            K=K,
            A_dtype="float16",
            W_dtype=QUANTIZED_DTYPE[bitwidth],
            accum_dtype="float16",
            out_dtype="float16",
            with_bias=False,
            group_size=qweight.shape[1] * 2,
            with_scaling=True,
            with_zeros=True,
            zeros_mode="quantized",
        )
        matmul = bitblas.Matmul(config=matmul_config, from_database=True)
        global_matmul_registry[f'{bitwidth},{M},{N},{K}'] = matmul
    output_tensor = matmul(x, qweight, scale=scale, zeros=qzero)
    return output_tensor
