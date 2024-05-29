import torch

def ibmm_fp16(
        indices, metas, y, x, qweight, 
        scale, g_idx=None, bias=None
    ):
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices, counts = torch.unique(valid_indices, sorted=False, return_counts=True)
    for id, count in zip(unique_indices, counts):
        idx_mask = indices == id
        inp = x[idx_mask]
        output = torch.matmul(inp, qweight[id].T)
        y[idx_mask] += output[:count]
    return y