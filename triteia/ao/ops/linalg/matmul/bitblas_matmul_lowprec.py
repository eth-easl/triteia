import bitblas
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE
from triteia.ao.utils.bitblas_utils import get_or_create_bitblas_operator

def bitblas_quant_mm_248(bitwidth, x, qweight, qzero, scale, g_idx=None, bias=None):
    assert qweight.shape[1]*2 == x.shape[1], f"2*qweight.shape[1] != x.shape[1], got {qweight.shape[1]*2} != {x.shape[1]}"
    assert qweight.shape[0] == qzero.shape[0]*2, f"qweight.shape[0] != qzero.shape[0], got {qweight.shape[0]} != {qzero.shape[0]*2}"
    assert qzero.shape[0] * 2 == scale.shape[0], f"qzero.shape[1] * 2 != scale.shape[0], got {qzero.shape[1] * 2} != {scale.shape[0]}"

    M = x.shape[0]
    N = qweight.shape[0] #   outfeatures
    K = qweight.shape[1] * 2 # infeatures

    matmul_config = bitblas.MatmulConfig(
        M=M,
        N=N,
        K=K,
        fast_decoding=True,
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
    matmul = get_or_create_bitblas_operator(matmul_config)
    output_tensor = matmul(x, qweight, scale=scale, zeros=qzero)
    return output_tensor
