import torch
from triteia.ao.utils.gen_tensor import gen_pruned_quant4_NT

def generate_2_4_pruned(num_models, m, n, groupsize=-1, device="cuda:0"):
    refs = []
    qs = []
    scales = []
    for i in range(num_models):
        ref, q, s = gen_pruned_quant4_NT(m, n, groupsize=groupsize, DEV=device)
        refs.append(ref)
        qs.append(q)
        scales.append(s)
    return torch.stack(refs), torch.stack(qs), torch.stack(scales)