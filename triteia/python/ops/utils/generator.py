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
    layer = sparse_low_precision_linear(m, k, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = m
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
