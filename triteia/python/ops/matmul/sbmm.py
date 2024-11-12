import torch
from triteia.python.capi import sbmm_2_4
from triteia.python.capi import mul_2_4
from triteia.python.capi import sbmm_2_4_multilaunch


def sbmm_16bit_forloop(weights, xs, indices, base_weight=None):
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(xs.shape[0], weights.shape[2], dtype=xs.dtype, device=xs.device)
    if torch.all(indices == -1):
        return y

    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    workspace = torch.zeros(y.shape[1] // 128 * 16, device=xs.device, dtype=torch.int32)
    for id, count in zip(unique_indices, counts):
        if id != -1:
            idx_mask = indices == id
            inp = xs[idx_mask]
            output = torch.matmul(inp, weights[id])
            y[idx_mask] += output
    return y


def sbmm_4bit_2_4_forloop(qweights, xs, metas, ss, indices, base_weight=None):
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(xs.shape[0], ss.shape[2], dtype=xs.dtype, device=xs.device)
    if torch.all(indices == -1):
        return y
    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    workspace = torch.zeros(y.shape[1] // 128 * 16, device=xs.device, dtype=torch.int32)
    for id, count in zip(unique_indices, counts):
        if id != -1:
            idx_mask = indices == id
            inp = xs[idx_mask]
            output = torch.zeros(
                (inp.shape[0], y.shape[1]), dtype=torch.float16, device=xs.device
            )
            mul_2_4(
                inp,
                qweights[id],
                metas[id],
                output,
                ss[id],
                workspace,
            )
            y[idx_mask] += output
    return y


def sbmm_4bit_2_4_multilaunch(qweights, xs, metas, ss, indices, base_weight=None):
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(xs.shape[0], ss.shape[2], dtype=xs.dtype, device=xs.device)
    if torch.all(indices == -1):
        return y
    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    first_nonnegative = torch.where(indices != -1)[0][0]
    if first_nonnegative > 0:
        unique_indices = unique_indices[1:]
        counts = counts[1:]
    start = torch.cat(
        (
            torch.tensor([first_nonnegative]).cuda(),
            (torch.cumsum(counts, dim=0) + first_nonnegative)[:-1],
        )
    )
    workspace = torch.zeros(len(unique_indices), y.shape[1] // 8, device=xs.device)
    output = torch.zeros_like(y)
    sbmm_2_4_multilaunch(
        xs,
        qweights,
        metas,
        output,
        ss,
        unique_indices,
        workspace,
        start,
        counts,
    )
    y += output
    return y


def sbmm_4bit_2_4_native(qweights, xs, metas, ss, indices, base_weight=None):
    if base_weight is not None:
        y = torch.matmul(xs, base_weight.t())
    else:
        y = torch.zeros(xs.shape[0], ss.shape[2], dtype=xs.dtype, device=xs.device)
    if torch.all(indices == -1):
        return y
    unique_indices, counts = torch.unique_consecutive(indices, return_counts=True)
    if len(unique_indices) == 1:
        workspace = torch.zeros(
            y.shape[1] // 128 * 16, device=xs.device, dtype=torch.int32
        )
        output = torch.zeros_like(y)
        mul_2_4(
            xs,
            qweights[unique_indices[0]],
            metas[unique_indices[0]],
            output,
            ss[unique_indices[0]],
            workspace,
        )
        y += output
    else:
        unique_indices = unique_indices.int()
        counts = counts.int()
        first_nonnegative = torch.where(indices != -1)[0][0]
        if first_nonnegative > 0:
            unique_indices = unique_indices[1:]
            counts = counts[1:]
        start = torch.cat(
            (
                torch.tensor([first_nonnegative]).cuda(),
                (torch.cumsum(counts, dim=0) + first_nonnegative)[:-1],
            )
        ).int()
        workspace = torch.zeros(
            len(unique_indices), y.shape[1] // 8, device=xs.device, dtype=torch.int32
        )
        output = torch.zeros(
            (xs.shape[0], ss.shape[2]), dtype=torch.float16, device=xs.device
        )
        sbmm_2_4(
            xs,
            qweights,
            metas,
            output,
            ss,
            unique_indices,
            workspace,
            start,
            counts,
        )
        y += output
    return y
