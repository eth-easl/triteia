import torch

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