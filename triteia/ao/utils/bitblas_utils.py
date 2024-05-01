import os
import bitblas
from fractions import Fraction
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE
from bitblas.cache.operator import global_operator_cache
from bitblas import auto_detect_nvidia_target

if "BITBLAS_TARGET" not in os.environ:
    BITBLAS_TARGET = auto_detect_nvidia_target()
else:
    BITBLAS_TARGET = os.environ["BITBLAS_TARGET"]
BITBLAS_DATABASE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "bitblas")
global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)

def convert_to_bitblas(bitwidth, module_name, tensors, zeros_mode="quantized"):
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

    from triteia.ao.nn.linear_bitblas import Linear as BitblasLinear
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
        zeros_mode=zeros_mode,
        enable_tuning=False,
    )
    bitblas_linear.repack_from_weights(qweight, scales, qzero, bias)
    return (
        bitblas_linear.qweight,
        bitblas_linear.scales,
        bitblas_linear.zeros,
        bitblas_linear.bias,
    )

def get_or_create_bitblas_operator(config, enable_tuning=True):
    bitblas_matmul = global_operator_cache.get(config)
    if bitblas_matmul is None:
        print("BitBLAS Operator not found in global_operator_cache, creating...")
        # don't tune it here so we can pass parameters
        bitblas_matmul = bitblas.Matmul(
            config=config, 
            target=BITBLAS_TARGET,
            enable_tuning=False,
        )
        if enable_tuning:
            bitblas_matmul.hardware_aware_finetune(topk=20, parallel_build=True)
            global_operator_cache.add(config, bitblas_matmul)
            global_operator_cache.save_into_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )
            print(
                "BitBLAS Tuning done, appended operator to global_operator_cache."
            )
        else:
            print("BitBLAS Operator created without tuning, not supposed to be used unless you know what you're doing...")
    else:
        pass
    return bitblas_matmul

def dequant_zeros(bitwidth):
    pass