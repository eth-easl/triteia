import torch
import triteia.lib.marlin as marlin

def ibmm_sparse_marlin(bitwidth, indices,metas, y, x, qweight, scale, g_idx=None, bias=None):
    unique_indices, counts = torch.unique(indices, sorted=False, return_counts=True)
    for id, count in zip(unique_indices, counts):
        if id != -1:
            workspace = torch.zeros(y.shape[1] // 128 * 16, device=x.device)
            idx_mask = indices == id
            inp = x[idx_mask]
            output = torch.zeros((inp.shape[0], y.shape[1]), dtype=torch.float16, device=x.device)
            marlin.mul_2_4(
                inp,
                qweight[id],
                metas[id],
                output,
                scale[id],
                workspace,
            )
            y[idx_mask] = output
    return y

def ibmm_sparse_marlin_stream(bitwidth, indices,metas, y, x, qweight, scale, g_idx=None, bias=None):
    unique_indices, counts = torch.unique(indices, sorted=False, return_counts=True)
    first_nonnegative = torch.where(indices != -1)[0][0]
    if first_nonnegative > 0:
        unique_indices = unique_indices[1:]
        counts = counts[1:]
    start = torch.cat((torch.tensor([first_nonnegative]).cuda(), (torch.cumsum(counts, dim=0)+ first_nonnegative)[:-1]))
    workspace = torch.zeros(len(unique_indices), y.shape[1] // 8, device=x.device)
    
    marlin.mul_stream(
        x,
        qweight,
        metas,
        y,
        scale,
        unique_indices,
        workspace,
        start,
        counts,
    )
    return y

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