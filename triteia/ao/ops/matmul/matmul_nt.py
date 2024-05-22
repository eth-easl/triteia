import tvm
from tvm import te

def matmul_nt(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    if not isinstance(M, int):
        M = tvm.te.var("m")
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N, K), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype(accum_dtype) * B[j, k].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)