import torch
from ao.ops.linalg import bmm, matmul, native_matmul_lowprec_248


def default_compile(fn):
    return torch.compile(
        fn,
        mode="reduce-overhead",
        dynamic=True,
    )


# bmm = default_compile(bmm)
# matmul = default_compile(matmul)

__all__ = ["bmm", "matmul", "native_matmul_lowprec_248"]
