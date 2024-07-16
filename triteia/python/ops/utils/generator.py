import torch
import numpy as np
import torch.nn as nn
from .sparsity import mask_creator


def gen_sparse_quant4_NT(m, k, groupsize=-1, device="cuda", prune_n=2, prune_m=4):
    from triteia.python.nn.linear import sparse_low_precision_linear

    maxq = 2**4 - 1
    w = torch.randn((m, k), dtype=torch.half, device=device)
    k_sp = k // 2
    w = w.t()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, m))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, m))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, m)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)
    mask = mask_creator(w.T, n=prune_n, m=prune_m).cuda().bool()
    uncompress = (mask * ref.T).T
    s = s.reshape((-1, m)).contiguous()
    linear = nn.Linear(k, m)
    linear.weight.data = ref
    layer = sparse_low_precision_linear(k, m, groupsize=-1)
    if groupsize == -1:
        groupsize = k
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=device)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=device)
    layer.s = torch.empty(
        (k_sp // (groupsize // 2), m), dtype=torch.half, device=device
    )
    layer.pack(uncompress, s, True)
    q = layer.B
    s = layer.s
    meta = layer.meta
    return uncompress, q, s, meta


def gen_batched_sparse_quant4_NT(b, m, n, groupsize=-1, device="cuda:0"):
    metas = []
    qs = []
    scales = []
    uncompressed = []
    for i in range(b):
        unc, q, s, meta = gen_sparse_quant4_NT(m, n, groupsize=groupsize, device=device)
        uncompressed.append(unc)
        qs.append(q)
        scales.append(s)
        metas.append(meta)
    uncompressed = torch.stack(uncompressed).to(device)
    qs = torch.stack(qs).to(device)
    scales = torch.stack(scales).to(device)
    metas = torch.stack(metas).to(device)
    return uncompressed, qs, scales, metas


def generate_model_distribution(distribution, num_queries, num_models):
    to_eval_models = list(range(-1, num_models))
    if distribution == "uniform":
        models = np.random.choice(to_eval_models, num_queries)
    if distribution == "distinct":
        models = to_eval_models
    if distribution.startswith("zipf"):
        alpha = float(distribution.split(":")[1])
        assert alpha > 1, "alpha must be greater than 1"
        probs = [i**alpha for i in range(1, len(to_eval_models) + 1)]
        probs = np.array(probs) / sum(probs)
        models = np.random.choice(to_eval_models, num_queries, p=probs)
    return torch.tensor(models, dtype=torch.int32, device="cuda")


def fp16_to_sparse(weights, scale, device="cuda"):
    from triteia.python.nn.linear import sparse_low_precision_linear

    k, m = weights.shape
    k_sp = k // 2
    s = scale
    layer = sparse_low_precision_linear(k, m, groupsize=-1)
    groupsize = k
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=device)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=device)
    layer.s = torch.empty(
        (k_sp // (groupsize // 2), m), dtype=torch.half, device=device
    )
    layer.pack(weights, scale, True)
    q = layer.B
    s = layer.s
    meta = layer.meta
    return weights, q, s, meta, layer.meta_raw


@torch.no_grad()
def torch_weight_to_sparse_marlin(weight, scale, tp_size=1, chunk_by="column"):
    """
    Args:
        weight: torch.Tensor of shape (in_features, out_features)
        scale: torch.Tensor of shape (1, out_features)
        tp_size: tensor parallelism size
        chunk_by: "column" or "row"
    """
    from triteia.python.nn.linear import sparse_low_precision_linear

    assert chunk_by in ["column", "row"], "chunk_by must be either 'column' or 'row'"
    assert weight.dim() == 2, "weight must be a 2D tensor"
    assert weight.size(0) % tp_size == 0, "out_features must be divisible by tp_size"
    assert weight.size(1) == scale.size(
        1
    ), "out_features of weight and scale must match"

    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()

    qweights, scales, metas = [], [], []
    for i in range(tp_size):
        if chunk_by == "column":
            tp_weight = weight[
                :, i * weight.size(1) // tp_size : (i + 1) * weight.size(1) // tp_size
            ]
            tp_scales = scale[
                :, i * scale.size(1) // tp_size : (i + 1) * scale.size(1) // tp_size
            ]
        elif chunk_by == "row":
            tp_weight = weight[
                i * weight.size(0) // tp_size : (i + 1) * weight.size(0) // tp_size, :
            ]
            tp_scales = scale
        layer = sparse_low_precision_linear(
            infeatures=tp_weight.size(0), outfeatures=tp_weight.size(1), groupsize=-1
        )
        k, m = tp_weight.size(0), tp_weight.size(1)
        k_sp = k // 2
        layer.groupsize = k
        layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int)
        layer.meta = torch.empty((m, k // 16), dtype=torch.int16)
        layer.s = torch.empty((k_sp // (k // 2), m), dtype=torch.half)
        layer.pack(tp_weight, scales=tp_scales, trans=True)
        qweights.append(layer.B.cuda().contiguous())
        scales.append(layer.s.cuda().contiguous())
        metas.append(layer.meta.cuda().contiguous())
    return qweights, scales, metas
