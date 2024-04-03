import torch
from ao.ops.linalg import bmm, matmul


def default_compile(fn):
    return torch.compile(
        fn,
        mode="reduce-overhead",
        # dynamic=True,
    )


# bmm = default_compile(bmm)
# matmul = default_compile(matmul)

__all__ = ["bmm", "matmul"]
