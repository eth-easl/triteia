import torch
from triteia.python.capi import add_lora_sgmv_cutlass

def lora_forloop(weights_A, weights_B, xs, indices, base_weight=None):
    
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(xs.shape[0], weights_B.shape[2], dtype=xs.dtype, device=xs.device)
    if torch.all(indices == -1):
        return y

    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    for id, count in zip(unique_indices, counts):
        if id != -1:
            idx_mask = indices == id
            inp = xs[idx_mask]
            output = torch.matmul(torch.matmul(inp, weights_A[id]), weights_B[id])
            y[idx_mask] += output
    return y

def lora_sgmv(weights_A, weights_B, xs, indices, base_weight=None):
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(xs.shape[0], weights_B.shape[2], dtype=xs.dtype, device=xs.device)
    if torch.all(indices == -1):
        return y
    
    lora_rank = weights_A.shape[2]
    layer_idx = 0
    #indices.append(indices.xs.shape[0])
    print(indices)
    s = indices
    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    s = torch.cat((unique_indices, torch.tensor([xs.shape[0]], device = xs.device, dtype=torch.int32)))
    print(s.dtype)
    print(y.dtype)
    print(weights_A.dtype)
    print(weights_B.dtype)
    print(lora_rank)
    
    wa_ptr = torch.tensor([t.data_ptr() for t in weights_A], dtype=torch.int64, device=xs.device)
    wb_ptr = torch.tensor([t.data_ptr() for t in weights_B], dtype=torch.int64, device=xs.device)
    add_lora_sgmv_cutlass(y, xs, wa_ptr, wb_ptr, s, layer_idx, lora_rank)

    return y