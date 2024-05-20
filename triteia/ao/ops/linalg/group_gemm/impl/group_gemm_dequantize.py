# pre-transformed tir expression of group_gemm
import tvm
from tvm import te
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
    if not isinstance(M, int):
        M = tvm.te.var("m")
    if with_bias:
        raise ValueError("with_bias is not supported")

    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K

    indices = te.placeholder((M,), dtype="int32", name="indices")

    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((num_models, N, K // storage_nbit * bit), name="B", dtype=storage_dtype)
    Y = te.placeholder((M, N), name="Y", dtype=accum_dtype)
    Scale = te.placeholder((num_models, N, K // group_size), name="Scale", dtype=in_dtype)
    Zeros = te.placeholder((num_models, N, K // group_size), name="Zeros", dtype=in_dtype)
    QZeros = te.placeholder((num_models, (K // group_size), N // storage_nbit * bit),name="QZeros",dtype=storage_dtype)

    def qzeros_dequantize(b, k, n):
        return _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
            bit,
            QZeros[b, k, n // n_float_per_elem],
            n % n_float_per_elem,
            dtype=storage_dtype,
        )

    Dequantize_qzeros = te.compute(
        (num_models, K // group_size, N),
        qzeros_dequantize,
        name="Dequantize_zeros",
    )
    
    def decode_func(n_m, n, k):
        if with_zeros and zeros_mode == "quantized":
            w = _tir_packed_to_unsigned_convert_with_zeros(storage_type, storage_nbit)(
                bit,
                B[n_m, n, k // n_float_per_elem],
                k % n_float_per_elem,
                Dequantize_qzeros[n_m, k // group_size, n],
                dtype=in_dtype,
            )
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))
        if not with_scaling:
            return w
        if not with_zeros:
            return w * Scale[n_m, n, k // group_size]
        if zeros_mode == "original":
            w = (w - Zeros[n_m, n, k // group_size]) * Scale[n, k // group_size]
        elif zeros_mode == "rescale":
            w = w * Scale[n_m, n, k // group_size] - Zeros[n, k // group_size]
        elif zeros_mode == "quantized":
            w = w * Scale[n_m, n, k // group_size]
        else:
            raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))
        return w

    B_decode = te.compute((num_models, N, K), decode_func, name="B_decode")

    # k = te.reduce_axis((0, K), name="k")
    # C = te.compute(
    #     (num_reqs, N),
    #     lambda i, j: te.sum(
    #         A[i, k].astype(accum_dtype) * B_decode[indices[i], k, j].astype(accum_dtype), axis=k
    #     ),
    #     name="C"
    # )
    # D = te.compute(
    #     (num_reqs, N),
    #     lambda i, j: C[i, j] + Y[i, j],
    #     name="D",
    # )
    # E = te.compute((num_reqs, N), lambda i, j: D[i, j].astype(out_dtype), name="E")

    args = [indices, A, B, Y]
    last_output = Y
    if with_scaling:
        args.append(Scale)

    if with_zeros:
        if zeros_mode == "quantized":
            args.append(QZeros)
        else:
            args.append(Zeros)
    args.append(Y)

    for idx, arg in enumerate(args):
        print(f"Arg {idx}: Type={type(arg)}, Value={arg}")

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
    # print(func.script())
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
    