import torch
from triteia.ao.ops.linalg.matmul.bmm_lowprec import quant_bmm_248

def quant_select_bmm_248(bitwidth, indices,y, x, qweight, qzero, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    y_indices = torch.arange(y.shape[0], device=y.device)[mask]
    x = x[mask, :].unsqueeze(1)
    valid_qweights = qweight.index_select(0, valid_indices)
    valid_qzeros = qzero.index_select(0, valid_indices)
    valid_scales = scale.index_select(0, valid_indices)
    valid_g_idx = g_idx.index_select(0, valid_indices)
    output = quant_bmm_248(bitwidth, x, valid_qweights, valid_qzeros, valid_scales, valid_g_idx, bias=bias).squeeze(1)
    y.index_add_(0, y_indices, output)
    return y