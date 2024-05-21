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
    N = qweight.shape[1]
    K = qweight.shape[2] // pack_factor
    group_mm_config = GroupMatmulWeightOnlyDequantizeConfig(
        M=(1,2,4,8,16),
        N=N,
        K=K,
        num_models=4,
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
    output_tensor = group_mm(x, qweight, indices, scale=scale, zeros=qzero)
    return output_tensor


@torch.inference_mode()
def group_gemm_2(bitwidth, indices, y, x, qweight, qzero, scale):
    pack_factor = Fraction(bitwidth, DTYPES_BIT[BITBLAS_STORAGE_DTYPE])
    N = qweight.shape[1]
    K = qweight.shape[2] // pack_factor
    
    mask = indices != -1
    valid_indices = indices[mask]
    y_indices = torch.arange(y.shape[0], device=y.device)[mask]
    x = x[mask, :]
    valid_qweights = qweight.index_select(0, valid_indices)
    valid_qzeros = qzero.index_select(0, valid_indices)
    valid_scales = scale.index_select(0, valid_indices)
    # print(f"x.shape: {x.shape} valid qweights shape: {valid_qweights.shape}, valid_qzeros shape: {valid_qzeros.shape}, valid_scales shape: {valid_scales.shape}")
    # group_mm_config = GroupMatmulWeightOnlyDequantizeConfig(
    #     M=x.shape[0],
    #     N=N,
    #     K=K,
    #     num_models=x.shape[0],
    #     fast_decoding=True,
    #     in_dtype="float16",
    #     bit=bitwidth,
    #     accum_dtype="float16",
    #     out_dtype="float16",
    #     source_format="uint",
    #     layout="nt",
    #     with_bias=False,
    #     group_size=K,
    #     with_scaling=True,
    #     with_zeros=True,
    #     zeros_mode="quantized",
    # )
    # group_mm = get_or_create_bitblas_operator(group_mm_config, type="group_gemm")
    # group_mm(
    #     x, qweight, indices, scale=scale, zeros=qzero, output=y
    # )
    return y