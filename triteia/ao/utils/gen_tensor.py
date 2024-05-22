import torch
import marlin
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)
from triteia.ao.nn.linear_bitblas import Linear as BitblasLinear
from marlin._semi_structured_conversions import (
    mask_creator
)
def gen_quant4(m, n, groupsize=-1, DEV="cuda:0"):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
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
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s

def gen_pruned_quant4_NT(m, k, groupsize=-1, DEV="cuda:0"):
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
    linear = nn.Linear(k, m)
    linear.weight.data = ref

    layer = marlin.Layer_2_4(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = m
    layer.groupsize = groupsize
    layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=DEV)
    layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
    layer.s = torch.empty((k_sp // (groupsize // 2), m), dtype=torch.half, device=DEV)
    layer.pack(linear, s, True)
    q = layer.B
    s = layer.s
    meta = layer.meta

    return uncompress, q, s, meta

def gen_quant(bitwidth, k, n, groupsize=-1):
    maxq = 2**bitwidth
    w = torch.randn((k, n), dtype=torch.half, device="cpu")
    original_w = w.clone()
    if groupsize == -1:
        groupsize = k
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    # Quantize.
    w = torch.round(w / s).int()
    # Unsigned storage.
    w += (maxq) // 2
    w = torch.clamp(w, 0, maxq)
    # Dequantize.
    ref = (w - (maxq) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()
    return original_w, linear, s, (w - (maxq) // 2)

def generate_quantized_weight(bitwidth, k, n, group_size):
    in_features = k
    out_features = n
    original_w, linear, s, qw = gen_quant(
        bitwidth, in_features, out_features, group_size
    )
    cuda_old_linear = CudaOldQuantLinear(
        bits=bitwidth,
        group_size=in_features,
        infeatures=in_features,
        outfeatures=out_features,
        bias=False,
    )
    max_zero_int = 2*bitwidth - 1
    zeros = torch.full((in_features // group_size, out_features), max_zero_int, dtype=torch.int32)
    cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)
    return cuda_old_linear.qweight, cuda_old_linear.scales, cuda_old_linear.qzeros

def generate_bitblas_weight(bitwidth, k, n, group_size):
    qweight, scales, qzero = generate_quantized_weight(bitwidth, k, n, group_size)
    bitblas_linear = BitblasLinear(
        in_features=k,
        out_features=n,
        bias=False,
        A_dtype="float16",
        W_dtype=f"uint{bitwidth}",
        accum_dtype="float16",
        out_dtype="float16",
        group_size=group_size,
        with_scaling=True,
        with_zeros=True,
        zeros_mode="quantized",
        enable_tuning=False,
    )
    bitblas_linear.repack_from_weights(qweight, scales, qzero, None)
    return bitblas_linear.qweight.cuda(), bitblas_linear.scales.cuda(), bitblas_linear.zeros.T.cuda()