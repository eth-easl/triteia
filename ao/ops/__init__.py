import torch
from ao.linalg import bmm, matmul

def default_compile(fn):
    return torch.compile(fn, 
        mode="reduce-overhead"
    )

bmm = default_compile(bmm)
matmul = default_compile(matmul)

__all__ = [
    'bmm',
    'matmul'
]