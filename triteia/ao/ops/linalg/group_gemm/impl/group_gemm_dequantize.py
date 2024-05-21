# pre-transformed tir expression of group_gemm
import tvm
from tvm import te, topi
from bitblas.quantization import (
    _tir_packed_int_to_int_convert,
    _tir_packed_to_signed_convert,
    _tir_packed_to_unsigned_convert,
    _tir_u32_to_f4_to_f16,
    _tir_packed_to_unsigned_convert_with_zeros,
)
import logging
logger = logging.getLogger(__name__)

def group_matmul_nt_dequantize_b(
    M,
    N,
    K,
    num_models,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    zeros_mode="original",
):
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K
    
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((num_models, N, K // storage_nbit * bit), name="B", dtype=storage_dtype)
    Indices = te.placeholder((M, ), name="Indices", dtype="int32")
    Scale = te.placeholder((num_models, N, K // group_size), name="Scale", dtype=in_dtype)
    Zeros = te.placeholder((num_models, N, K // group_size), name="Zeros", dtype=in_dtype)
    QZeros = te.placeholder((num_models, (K // group_size), N // storage_nbit * bit),name="QZeros", dtype=storage_dtype)
    
    Y = te.placeholder((M, N), name="Y", dtype=out_dtype)

    def qzeros_dequantize(i, k, n):
        return _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
            bit,
            QZeros[i, k, n // n_float_per_elem],
            n % n_float_per_elem,
            dtype=storage_dtype,
        )

    Dequantize_qzeros = te.compute(
        (num_models, K // group_size, N),
        qzeros_dequantize,
        name="Dequantize_zeros",
    )
    
    def decode_func(i, n, k):
        if with_zeros and zeros_mode == "quantized":
            w = _tir_packed_to_unsigned_convert_with_zeros(storage_type, storage_nbit)(
                bit,
                B[i, n, k // n_float_per_elem],
                k % n_float_per_elem,
                Dequantize_qzeros[i, k // group_size, n],
                dtype=in_dtype,
            )
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

        if not with_scaling:
            return w

        if not with_zeros:
            return w * Scale[i, n, k // group_size]

        if zeros_mode == "original":
            w = (w - Zeros[i, n, k // group_size]) * Scale[n, k // group_size]
        elif zeros_mode == "rescale":
            w = w * Scale[i, n, k // group_size] - Zeros[n, k // group_size]
        elif zeros_mode == "quantized":
            w = w * Scale[i, n, k // group_size]
        else:
            raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))
        return w
    
    B_decode = te.compute((num_models, N, K), decode_func, name="B_decode")
    
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B_decode[Indices[i], j, k].astype(accum_dtype), axis=k),
        name="C",
    )
    D = te.compute((M, N), lambda i, j: (C[i, j]).astype(out_dtype), name="D")
    
    args = [A, B, Indices]
    last_output = D
    if with_scaling:
        args.append(Scale)
    if with_zeros:
        if zeros_mode == "quantized":
            args.append(QZeros)
        else:
            args.append(Zeros)
    args.append(last_output)
    
    func = te.create_prim_func(args).with_attr(
        "dequantize_info",
        {
            "B_decode": {
                "decode_block": "B_decode",
                "fast_decoding": fast_decoding,
                "source_format": {
                    "bits": bit,
                    "format": source_format,
                },
                "storage_dtype": storage_dtype,
                "target_format": in_dtype,
                "with_scaling": with_scaling,
                "with_zeros": with_zeros,
                "zeros_mode": zeros_mode,
                "group_size": group_size,
            }
        },
    )
    return tvm.IRModule.from_expr(func)

def select_implementation(
    M=None,
    N=1024,
    K=1024,
    num_models=32,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    layout="nt",
    zeros_mode="original",
    propagate_a=False,
    propagate_b=False,
):
    return group_matmul_nt_dequantize_b(
        M,
        N,
        K,
        num_models,
        in_dtype,
        out_dtype,
        accum_dtype,
        bit,
        storage_dtype,
        source_format,
        with_scaling,
        with_zeros,
        group_size,
        fast_decoding,
        with_bias,
        zeros_mode,
    )
    