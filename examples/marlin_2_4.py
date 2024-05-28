import torch
import torch.nn as nn
from triteia.lib.marlin import Layer_2_4, mask_creator, mul_2_4

DEV = torch.device("cuda:0")

def quant_4_nt(w,s, groupsize=-1):
    m = w.shape[0]
    k = w.shape[1]
    maxq = 2**4 - 1
    k_sp = k // 2
    w = w.t()
    if groupsize != -1:
        w = w.reshape((-1, groupsize, m))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
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
    mask = mask_creator(w.T).cuda().bool()
    uncompress = ref
    s = s.reshape((-1, m)).contiguous()
    layer = Layer_2_4(
        ref.shape[1],
        ref.shape[0],
        groupsize=groupsize
    )
    if groupsize == -1:
        groupsize = k
    layer.n = m
    layer.k = k
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=DEV)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
    layer.s = torch.empty((k_sp // (groupsize // 2), m), dtype=torch.half, device=DEV)
    print(f"ref: {ref.shape}, s: {s.shape}")
    layer.pack(ref, s, True)
    q = layer.B
    s = layer.s
    meta = layer.meta
    return uncompress, q, s, meta

if __name__=="__main__":
    k = 5632
    m = 2048
    n = 1
    groupsize = -1
    workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)
    A = torch.randn((n, k), dtype=torch.half, device=DEV)
    fp16_weight = torch.randn((m, k), dtype=torch.half, device=DEV)
    # mask fp16_weight
    mask = mask_creator(fp16_weight.T).cuda().bool()
    fp16_weight = (mask * fp16_weight.T).T

    s = torch.max(torch.abs(fp16_weight.t()), 0, keepdim=True)[0]
    maxq = 2 ** 4 - 1
    s *= 2 / maxq

    print(f"fp16_weight: {fp16_weight.shape}, s: {s.shape}")
    print(f"fp16_weight: {fp16_weight}, s: {s}")

    B_ref, B, s, meta = quant_4_nt(fp16_weight, s, groupsize=groupsize)
    
    C_ref = torch.matmul(A, B_ref)

    layer = Layer_2_4(k, m, groupsize =-1)

    layer.B = B
    layer.meta = meta
    layer.s = s
    layer.workspace = workspace
    print(f"layer.B: {layer.B.shape}, layer.s: {layer.s.shape}, layer.meta: {layer.meta.shape}")
    C = layer(A)
    torch.cuda.synchronize()

    print(f"max diff: {(C-C_ref).max()}")