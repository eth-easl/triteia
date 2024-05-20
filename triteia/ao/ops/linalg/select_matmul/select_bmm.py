import torch
from triteia.ao.ops.linalg.matmul.bmm_lowprec import quant_bmm_248
from triteia.ao.ops.linalg.matmul.bmm_lowprec import bitblas_loop_quant_bmm_248
from triteia.ao.ops.linalg.matmul.native_mm_lowprec import native_bmm_lowprec
from triteia.ao.ops.linalg.matmul.matmul_lowprec import quant_matmul_248_bitblas



def naive_quant_select_bmm_248(bitwidth, indices,y, x, qweight, qzero, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    y_indices = torch.arange(y.shape[0], device=y.device)[mask]
    x = x[mask, :].unsqueeze(1)
    valid_qweights = qweight.index_select(0, valid_indices)
    valid_qzeros = qzero.index_select(0, valid_indices)
    valid_scales = scale.index_select(0, valid_indices)
    valid_g_idx = g_idx.index_select(0, valid_indices)
    output = native_bmm_lowprec(bitwidth, x, valid_qweights, valid_qzeros, valid_scales, valid_g_idx, bias=bias).squeeze(1)
    y.index_add_(0, y_indices, output)
    return y

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

def bitblas_quant_select_bmm_248(bitwidth, indices,y, x, qweight, qzero, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    y_indices = torch.arange(y.shape[0], device=y.device)[mask]
    x = x[mask, :].unsqueeze(1)
    valid_qweights = qweight.index_select(0, valid_indices)
    valid_qzeros = qzero.index_select(0, valid_indices)
    valid_scales = scale.index_select(0, valid_indices)
    output = bitblas_loop_quant_bmm_248(bitwidth, x, valid_qweights, valid_qzeros, valid_scales, g_idx=g_idx, bias=bias).squeeze(1)
    y.index_add_(0, y_indices, output)
    return y

def ibmm(bitwidth, indices, y, x, qweight, qzero, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    # weight.shape: (max_deltas, outfeatures, infeatures)
    unique_indices = torch.unique(valid_indices)
    for id in unique_indices:
        idx_mask = indices == id
        inp = x[idx_mask]
        output = quant_matmul_248_bitblas(
            bitwidth, 
            inp, 
            qweight[id],
            qzero[id],
            scale[id],
        )
        y[idx_mask] += output
    return y


def vectorized_ibmm(bitwidth, indices, y, x, qweight, qzero, scale, g_idx=None, bias=None):
    mask = indices != -1
    valid_indices = indices[mask]
    # weight.shape: (max_deltas, outfeatures, infeatures)
    unique_indices = torch.unique(valid_indices)
    streams = []
    for id in unique_indices:
        stream = torch.cuda.Stream()
        streams.append(stream)
        with torch.cuda.stream(stream):
            idx_mask = indices == id
            inp = x[idx_mask]
            output = quant_matmul_248_bitblas(
                bitwidth, 
                inp, 
                qweight[id],
                qzero[id],
                scale[id],
            )
            y[idx_mask] += output
    # [torch.cuda.current_stream().wait_stream(stream) for stream in streams]
    # torch.cuda.synchronize()
    return y