import torch
from triteia.python.capi import mul_2_4


def matmul_4bit_2_4(qweight, x, meta, s):
    """Low precision sparse matrix multiplication with 2:4 sparsity.
    ----
    Parameters:
        A: `torch.Tensor` weight matrix of shape `(m, k)` in 4-bit format. (k//16, m*2)
        B: `torch.Tensor` input matrix of shape `(n, k/2)`
        meta: `torch.int` metadata information for 2:4 sparsity
        s: `torch.half` scales of shape `(n / groupsize /2, m)`
    """
    C = torch.zeros((x.shape[:-1] + (s.shape[1],)), dtype=x.dtype, device=x.device)
    workspace = torch.zeros(s.shape[1], dtype=torch.int32, device=x.device)
    mul_2_4(
        x,
        qweight,
        meta,
        C,
        s,
        workspace,
    )
    return C
