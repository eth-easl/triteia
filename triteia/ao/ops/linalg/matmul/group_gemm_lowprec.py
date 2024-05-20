import torch
from fractions import Fraction
try:
    import bitblas
    from triteia.ao.utils.dtypes import QUANTIZED_DTYPE
except ImportError:
    print("BitBlas not installed")
    
from triteia.ao.utils.dtypes import BITBLAS_DTYPES
from triteia.ao.utils.bitblas_utils import get_or_create_bitblas_operator
from fractions import Fraction
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE, BITBLAS_STORAGE_DTYPE, DTYPES_BIT
from triteia.ao.ops.linalg.group_gemm import GroupMatmulWeightOnlyDequantize, GroupMatmulWeightOnlyDequantizeConfig

@torch.inference_mode()
def group_gemm(bitwidth, indices, y, x, qweight, qzero, scale):
    
    pack_factor = Fraction(bitwidth, DTYPES_BIT[BITBLAS_STORAGE_DTYPE])
    assert x.shape[0] == y.shape[0], f"x.shape[0] != y.shape[0], got {x.shape[0]} != {y.shape[0]}"
    
    assert qweight.shape[2] // pack_factor == x.shape[1], f"qweight.shape[2] // pack_factor != x.shape[1], got {qweight.shape[2]//pack_factor} != {x.shape[1]}"
    assert qweight.shape[1] == qzero.shape[1] // pack_factor, f"qweight.shape[1] != qzero.shape[1], got {qweight.shape[1]} != {qzero.shape[1]//pack_factor}"
    
    assert qzero.shape[1] // pack_factor == scale.shape[1], f"qzero.shape[1] // pack_factor != scale.shape[1], got {qzero.shape[1] // pack_factor} != {scale.shape[1]}"
    
    assert x.device==qweight.device==qzero.device==scale.device, f"x.device != qweight.device != qzero.device != scale.device, got {x.device} != {qweight.device} != {qzero.device} != {scale.device}"
    opt_M = [1,2,4,8, 16]
    N = qweight.shape[1]
    K = qweight.shape[2] // pack_factor
    group_mm_config = GroupMatmulWeightOnlyDequantizeConfig(
        M=x.shape[0],
        N=N,
        K=K,
        num_models=qweight.shape[0],
        fast_decoding=True,
        in_dtype="float16",
        bit=bitwidth,
        accum_dtype="float16",
        out_dtype="float16",
        source_format="uint",
        layout="nt",
        with_bias=False,
        group_size=K,
        with_scaling=True,
        with_zeros=True,
        zeros_mode="quantized",
    )
    group_mm = get_or_create_bitblas_operator(group_mm_config, type="group_gemm")
    output_tensor = group_mm(x, qweight,indices, scale=scale, zeros=qzero)
    return output_tensor