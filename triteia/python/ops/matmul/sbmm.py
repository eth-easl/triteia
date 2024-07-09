import torch
from triteia.python.capi import sbmm_2_4
from .sparse_low_precision import matmul_4bit_2_4

def sbmm_4bit_2_4_forloop(qweights, xs, metas, ss, indices, base_weight=None):
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(
            xs.shape[0], ss.shape[2], dtype=xs.dtype, device=xs.device
        )
    if torch.all(indices==-1):
        return y

    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    first_nonnegative = torch.where(indices != -1)[0][0]
    if first_nonnegative > 0:
        unique_indices = unique_indices[1:]
        counts = counts[1:]
    start = torch.cat((torch.tensor([first_nonnegative]).cuda(), (torch.cumsum(counts, dim=0)+ first_nonnegative)[:-1]))
    workspace = torch.zeros(len(unique_indices), y.shape[1] // 8, device=xs.device)
    output = torch.zeros_like(y)