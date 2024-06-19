import torch

def ibmm_fp16(
        indices, metas, y, x, qweight, 
        scale, g_idx=None, bias=None
    ):
    if torch.all(indices == -1):
        return y
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices, counts = torch.unique(valid_indices, sorted=False, return_counts=True)
    for id, count in zip(unique_indices, counts):
        idx_mask = indices == id
        inp = x[idx_mask]
        output = torch.matmul(inp, qweight[id].T)
        y[idx_mask] += output
    return y

def ibmm_fp16_bmm(
        indices, metas, y, x, qweight
    ):
    if torch.all(indices == -1):
        return y
    mask = indices != -1
    valid_indices = indices[mask]
    x = x[mask, :].unsqueeze(1)
    valid_qweights = qweight.index_select(0, valid_indices)
    output = torch.bmm(x, valid_qweights).squeeze(1)
    y[mask] += output
    return y