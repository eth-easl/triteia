import torch
from triteia.ao.ops.linalg import (
    bmm,
    matmul,
    native_matmul_lowprec_248,
    quant_matmul_248,
    transpose_quant_matmul_248,
    native_bmm_lowprec,
    quant_bmm_248,
)
from triteia.ao.ops.activations.silu import silu


def default_compile(fn):
    return torch.compile(
        fn,
        mode="reduce-overhead",
        dynamic=True,
    )


# bmm = default_compile(bmm)
# matmul = default_compile(matmul)

__all__ = [
    "bmm",
    "matmul",
    "native_matmul_lowprec_248",
    "quant_matmul_248",
    "transpose_quant_matmul_248",
    "native_bmm_lowprec",
    "silu",
    "quant_bmm_248",
]
