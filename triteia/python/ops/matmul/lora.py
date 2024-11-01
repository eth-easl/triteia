import torch
from triteia.python.capi import add_lora_sgmv_cutlass, add_lora_bgmv

def lora_forloop(weights_A, weights_B, x, indices, base_weight=None):
    
    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    else:
        y = torch.zeros(x.shape[0], weights_B.shape[2], dtype=x.dtype, device=x.device)
    if torch.all(indices == -1):
        return y

    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    for id, count in zip(unique_indices, counts):
        if id != -1:
            idx_mask = indices == id
            inp = x[idx_mask]
            output = torch.matmul(torch.matmul(inp, weights_A[id]), weights_B[id])
            y[idx_mask] += output
    return y

def lora_bgmv(weights_A, weights_B, x, indices, base_weight=None, layer_idx=0, scale = 1.0):
    # need to transpose
    # need to seperate -1 and the rest
    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    else:
        y = torch.zeros(x.shape[0], weights_B.shape[2], dtype=x.dtype, device=x.device)
    if torch.all(indices == -1):
        return y

    # Transpose and ensure contiguity after transpose
    weights_A_T = weights_A.transpose(1, 2).contiguous()
    weights_B_T = weights_B.transpose(1, 2).contiguous()
    
    # Add layer dimension
    weights_A_T = weights_A_T.unsqueeze(1)
    weights_B_T = weights_B_T.unsqueeze(1)

    indices = indices.to(torch.long)

    mask = indices != -1
    inp_x = x[mask]
    inp_y = y[mask]
    inp_indices = indices[mask]
    add_lora_bgmv(inp_y, inp_x, weights_A_T, weights_B_T, inp_indices, layer_idx, scale)
    y[mask] = inp_y
    return y

def lora_sgmv(weights_A, weights_B, x, indices, base_weight=None, layer_idx=0):
    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    else:
        y = torch.zeros(x.shape[0], weights_B.shape[2], dtype=x.dtype, device=x.device)
    if torch.all(indices == -1):
        return y
    
    # get the rank from the weight matrix shape
    rank = weights_A.shape[2]
    
    # transform indices to sgvm format
    # s contains the starting index in x for each of the models
    s = [0]
    next = 0
    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    for i, idx in enumerate(unique_indices):
        # skip -1 models
        if idx == -1:
            continue
        # models that are not in unique_indices start and end at the same index
        for j in range(next, idx):
            s.append(s[-1])
        s.append(s[-1] + counts[i].item())
        next = idx + 1
    # clean up the missing models
    for i in range(next, weights_A.shape[0]):
        s.append(s[-1])
    s = torch.tensor(s, device = x.device, dtype = torch.int32)
    
    # remove the inputs for the -1 model
    input_x = x
    input_y = y
    remainder_y = torch.empty(0, device = x.device)
    if unique_indices[0].item() == -1:
        input_x = x[counts[0].item():]
        input_y = y[counts[0].item():]
        remainder_y = y[:counts[0].item()]


    # adding a layer dimension for each model
    weights_A = weights_A.unsqueeze(1)
    weights_B = weights_B.unsqueeze(1)

    # get the pointer to each model
    wa_ptr = torch.tensor([t.data_ptr() for t in weights_A], dtype=torch.int64, device=x.device)
    wb_ptr = torch.tensor([t.data_ptr() for t in weights_B], dtype=torch.int64, device=x.device)

    add_lora_sgmv_cutlass(input_y, input_x, wa_ptr, wb_ptr, s, layer_idx, rank)
    
    if unique_indices[0].item() == -1:
        output_y = torch.cat((remainder_y, input_y))
    else:
        output_y = input_y

    return output_y