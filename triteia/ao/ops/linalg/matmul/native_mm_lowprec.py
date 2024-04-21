import torch
from typing import Optional


def native_matmul_lowprec_248(
    bitwidth: int,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzero: torch.Tensor,
    scale: torch.Tensor,
    g_idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    assert x.dim() == 2, "x must be 2-dimensional"
    assert bitwidth in [2, 4, 8], "Only bitwidths of 2, 4, and 8 are supported"
    assert qweight.dtype == torch.int32, "qweight must be of type torch.int32"
    assert qweight.dim() == 2, "qweight must be 2-dimensional"

    infeatures = (
        qweight.shape[0] // bitwidth * 32
    )  # qweight is stored in 32-bit integers (packed)
    outfeatures = qweight.shape[1]

    assert qzero.dim() == 2, "qzero must be 2-dimensional"
    assert (
        qzero.shape[1] == outfeatures // 32 * bitwidth
    ), f"qzero has incorrect shape, should be (1, outfeatures// 32 * bitwidth). Expected (1, {infeatures // 32 * bitwidth, outfeatures}), got {qzero.shape}"
    assert (
        scale.shape[1] == outfeatures
    ), f"scale has incorrect shape, should be (1, outfeatures). Expected (1, {outfeatures}), got {scale.shape}"

    out_shape = x.shape[:-1] + (outfeatures,)
    wf = torch.tensor(list(range(0, 32, bitwidth)), dtype=torch.int32).unsqueeze(0)
    if wf.device != qzero.device:
        wf = wf.to(qzero.device)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzero, 2).expand(-1, -1, 32 // bitwidth),
        wf.unsqueeze(0),
    ).to(torch.int16 if bitwidth == 8 else torch.int8)
    torch.bitwise_and(zeros, (2**bitwidth) - 1, out=zeros)
    zeros = zeros + 1
    zeros = zeros.reshape(scale.shape)

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bitwidth, -1),
        wf.unsqueeze(-1),
    ).to(torch.int16 if bitwidth == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bitwidth) - 1, out=weight)

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    num_itr = g_idx.shape[0] // x.shape[-1]
    if num_itr == 1:
        weights = scale[g_idx.long()] * (weight - zeros[g_idx.long()])
    else:
        num_dim = g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = scale[:, i * num_dim : (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
            g_idx_i = g_idx[i * num_dim : (i + 1) * num_dim]
            weights.append(
                scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()])
            )
        weights = torch.cat(weights, dim=1)
    out = torch.matmul(x.half(), weights)
    out = out.reshape(out_shape)
    out = out + bias if bias is not None else out
    return out


def native_bmm_lowprec(
    bitwidth: int,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzero: torch.Tensor,
    scale: torch.Tensor,
    g_idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    assert x.dim() == 3, "x must be 3-dimensional (bsz, M, K)"
    # loop over the batch dimension
    out = []
    for i in range(x.shape[0]):
        out.append(
            native_matmul_lowprec_248(
                bitwidth,
                x[i],
                qweight[i],
                qzero[i],
                scale[i],
                g_idx[i],
                bias[i] if bias is not None else None,
            )
        )
    return torch.stack(out)
