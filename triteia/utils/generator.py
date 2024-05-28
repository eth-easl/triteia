import torch
from triteia.ao.utils.gen_tensor import gen_pruned_quant4_NT

def generate_2_4_pruned(num_models, m, n, groupsize=-1, device="cuda:0"):
    metas = []
    qs = []
    scales = []
    uncompressed = []
    for i in range(num_models):
        unc, q, s, meta = gen_pruned_quant4_NT(m, n, groupsize=groupsize, DEV=device)
        uncompressed.append(unc)
        qs.append(q)
        scales.append(s)
        metas.append(meta)
    uncompressed= torch.stack(uncompressed).to(device)
    qs = torch.stack(qs).to(device)
    scales = torch.stack(scales).to(device)
    metas = torch.stack(metas).to(device)
    return uncompressed, qs, scales, metas