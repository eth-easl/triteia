import torch
import triteia.lib.marlin as marlin

def ibmm_sparse_marlin(bitwidth, indices,metas, y, x, qweight, scale, g_idx=None, bias=None, base_weight=None,):
    # if all indices are -1, return y
    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    if torch.all(indices == -1):
        return y
    unique_indices, counts = torch.unique(indices, sorted=False, return_counts=True)
    for id, count in zip(unique_indices, counts):
        if id != -1:
            workspace = torch.zeros(y.shape[1] // 128 * 16, device=x.device)
            idx_mask = indices == id
            inp = x[idx_mask]
            output = torch.zeros((
                inp.shape[0], y.shape[1]
            ), dtype=torch.float16, device=x.device)
            marlin.mul_2_4(
                inp,
                qweight[id],
                metas[id],
                output,
                scale[id],
                workspace,
            )
            y[idx_mask] += output
    return y

def ibmm_sparse_marlin_stream(bitwidth, indices,metas, y, x, qweight, scale, g_idx=None, bias=None, base_weight=None, parallel=False):
    # if all indices are -1, return y
    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    if torch.all(indices == -1):
        return y
    unique_indices, counts = torch.unique_consecutive(
        indices, 
        return_counts=True
    )
    first_nonnegative = torch.where(indices != -1)[0][0]
    if first_nonnegative > 0:
        unique_indices = unique_indices[1:]
        counts = counts[1:]
    start = torch.cat((torch.tensor([first_nonnegative]).cuda(), (torch.cumsum(counts, dim=0)+ first_nonnegative)[:-1]))
    workspace = torch.zeros(len(unique_indices), y.shape[1] // 8, device=x.device)
    output = torch.zeros_like(y)
    marlin.mul_stream(
        x,
        qweight,
        metas,
        output,
        scale,
        unique_indices,
        workspace,
        start,
        counts,
    )
    torch.cuda.synchronize()
    y += output
    return y

def ibmm_native(bitwidth, indices,metas, y, x, qweight, scale, g_idx=None, bias=None, base_weight=None):
    # if all indices are -1, return y
    if x.shape[0] > 256:
        # if the number of requests is too large, use the naive version
        # it should be as good
        return ibmm_sparse_marlin_stream(bitwidth, indices,metas, y, x, qweight, scale, g_idx, bias, base_weight)
    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    if torch.all(indices == -1):
        return y
    unique_indices, counts = torch.unique_consecutive(
        indices, 
        return_counts=True
    )
    if len(unique_indices) == 1:
        workspace = torch.zeros(y.shape[1] // 128 * 16, device=x.device)
        output = torch.zeros_like(y)
        # no need to use ibmm, just a normal matmul
        marlin.mul_2_4(
            x,
            qweight[unique_indices[0]],
            metas[unique_indices[0]],
            output,
            scale[unique_indices[0]],
            workspace
        )
        y+= output
    else:
        unique_indices = unique_indices.int()
        counts = counts.int()
        first_nonnegative = torch.where(indices != -1)[0][0]
        if first_nonnegative > 0:
            unique_indices = unique_indices[1:]
            counts = counts[1:]
        start = torch.cat((torch.tensor([first_nonnegative]).cuda(), (torch.cumsum(counts, dim=0)+ first_nonnegative)[:-1])).int()
        workspace = torch.zeros(len(unique_indices), y.shape[1] // 8, device=x.device)
        output = torch.zeros((x.shape[0], y.shape[1]), dtype=torch.float16, device=x.device)
        marlin.ibmm_2_4(
            x,
            qweight,
            metas,
            output,
            scale,
            unique_indices,
            workspace,
            start,
            counts,
        )
        y += output
    torch.cuda.synchronize()
    return y