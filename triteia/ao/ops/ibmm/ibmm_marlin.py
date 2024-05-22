import torch
import marlin

def ibmm_sparse_marlin(bitwidth, indices,metas, y, x, qweight, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices = torch.unique(valid_indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        workspace = torch.zeros(y[idx_mask].shape[1] // 128 * 16, device=torch.device('cuda:0'))
        marlin.mul_2_4(
            inp,
            qweight[id],
            metas[id],
            y[idx_mask],
            scale[id],
            workspace,
        )
    return y

workspace = torch.zeros(4096 // 128 * 16, device=torch.device('cuda:0'))

def ibmm_marlin(bitwidth, indices, y, x, qweight, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices = torch.unique(valid_indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = torch.zeros_like(y[idx_mask], dtype=torch.float16, device=torch.device('cuda:0'))
        marlin.mul(
            inp,
            qweight[id],
            output,
            scale[id],
            workspace,
        )
        y[idx_mask] += output
    return y