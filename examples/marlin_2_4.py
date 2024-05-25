import torch
import torch.nn as nn
from triteia.lib.marlin import Layer_2_4, mask_creator, mul_2_4

DEV = torch.device("cuda:0")

def gen_quant4_NT(m, k, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((m, k), dtype=torch.half, device=DEV)
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
    mask = mask_creator(w.T).cuda().bool()
    uncompress = (mask * ref.T).T
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

k = 5632
m = 2048
n = 1
groupsize = -1
workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)

A = torch.randn((n, k), dtype=torch.half, device=DEV)
B_ref, B, s, meta = gen_quant4_NT(m, k, groupsize=groupsize)

C = torch.zeros((n, m), dtype=torch.half, device=DEV)
C_ref = torch.matmul(A, B_ref)

layer = Layer_2_4(k, m, groupsize =-1)

layer.B = B
layer.meta = meta
layer.s = s
layer.workspace = workspace
print(f"layer.B: {layer.B.shape}, layer.s: {layer.s.shape}, layer.meta: {layer.meta.shape}")
C = layer(A)
torch.cuda.synchronize()

print(f"Diff: {(C-C_ref).max()}")