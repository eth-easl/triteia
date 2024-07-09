import torch
from triteia.python.capi.marlin import bmm_2_4
from .sparse_low_precision import matmul_4bit_2_4


def bmm_4bit_2_4_forloop(qweights, xs, metas, ss):
    """
    Batched Low precision sparse matrix multiplcation with 2:4 sparsity.
    ----
    Parameters:

    """
    outputs = torch.zeros(
        (xs.shape[0], xs.shape[1], ss.shape[2]), dtype=xs.dtype, device=xs.device
    )
    for id in range(xs.shape[0]):
        outputs[id] = matmul_4bit_2_4(qweights[id], xs[id], metas[id], ss[id])
    return outputs


def bmm_4bit_2_4(qweights, xs, metas, ss):
    """
    Batched Low precision sparse matrix multiplcation with 2:4 sparsity.
    ----
    Parameters:

    """
    C = torch.zeros(
        (xs.shape[0], xs.shape[1], ss.shape[2]), dtype=xs.dtype, device=xs.device
    )
    workspace = torch.zeros(
        xs.shape[0] * max(ss.shape[2], ss.shape[1]), device=xs.device, dtype=torch.int32
    )
    bmm_2_4(xs, qweights, metas, C, ss, workspace)
    return C
