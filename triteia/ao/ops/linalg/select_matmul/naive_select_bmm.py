import torch
from triteia.ao.ops.linalg.matmul.bmm_lowprec import quant_bmm_248

def naive_quant_select_bmm_248(bitwidth, indices,y, x, qweight, qzero, scale, g_idx=None, bias=None):
    # x.shape: (batch_size, hidden_dim)
    # y.shape: (batch_size, hidden_dim)
    # qweight.shape: (num_deltas, hidden_dim, hidden_dim)
    bsz = qweight.shape[0]
    x = x.repeat(bsz, 1, 1)
    output = quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx, bias=None)
    for i in range(len(indices)):
        if indices[i] == -1:
            continue
        y[i,:] += output[indices[i], i, :]
    return y