import torch
from triteia.lib.marlin import Layer_2_4
from triteia.lib.marlin.semi_structured_conversions import mask_creator

def quant_4_nt(w,s, groupsize=-1, DEV="cuda:0"):
    m = w.shape[0]
    k = w.shape[1]
    w = w.t()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, m))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    layer = Layer_2_4(
        k,
        m,
        groupsize=-1
    )
    layer.n = m
    layer.k = k
    layer.groupsize = k if groupsize == -1 else groupsize
    layer.B = torch.empty((k//32, m * 2), dtype=torch.int, device=DEV)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
    layer.s = torch.empty((1, m), dtype=torch.half, device=DEV)
    print(f"ref: {w.shape}, s: {s.shape}")
    layer.pack(w, s, True)
    return layer.B, layer.s, layer.meta