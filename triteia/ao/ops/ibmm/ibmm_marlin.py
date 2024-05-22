import torch
import triteia.lib.marlin as marlin

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
    # sort x according to indices,
    # assuming indices is sorted
    # NOTE: double check in vllm!
    
    mask = indices != -1
    valid_indices = indices[mask]
    unique_indices, counts = torch.unique(valid_indices, sorted=False, return_counts=True)
    start = 0
    for id, count in zip(unique_indices, counts):
        inp_mask = indices == id
        inp = x[inp_mask]
        marlin.mul(
            inp,
            qweight[id],
            y[start:start+count,:],
            scale[id],
            workspace,
        )
        start += count
    return y

def ibmm_marlin_native(indices, y, x, qweight, scale):
    # convert indices to long type
    indices = indices.long()
    marlin.ibmm(x, qweight, y, scale, indices, workspace)