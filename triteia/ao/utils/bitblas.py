from fractions import Fraction
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE
from triteia.ao.ops.nn.linear_bitblas import Linear as BitblasLinear


def convert_to_bitblas(bitwidth, module_name, tensors):
    qweight = tensors[module_name + ".qweight"]
    qzero = tensors[module_name + ".qzeros"]
    scales = tensors[module_name + ".scales"]
    if module_name + ".bias" in tensors:
        bias = tensors[module_name + ".bias"]
    else:
        bias = None

    pack_factor = Fraction(bitwidth, 32)
    infeatures = qweight.shape[0] // pack_factor
    outfeatures = qweight.shape[1]
    group_size = infeatures

    bitblas_linear = BitblasLinear(
        in_features=infeatures,
        out_features=outfeatures,
        bias=False,
        A_dtype="float16",
        W_dtype=QUANTIZED_DTYPE[bitwidth],
        accum_dtype="float16",
        out_dtype="float16",
        group_size=group_size,
        with_scaling=True,
        with_zeros=True,
        zeros_mode="quantized",
    )
    bitblas_linear.repack_from_weights(qweight, scales, qzero, bias)
    return (
        bitblas_linear.qweight,
        bitblas_linear.scales,
        bitblas_linear.zeros,
        bitblas_linear.bias,
    )
