import ctypes
import torch
import triton
from typing import Optional
import triton.language as tl
import triteia.ao.utils.autotune as autotune
from fractions import Fraction
from torch.cuda.amp import custom_bwd, custom_fwd
try:
    import bitblas
    from triteia.ao.utils.dtypes import QUANTIZED_DTYPE
except ImportError:
    print("BitBlas not installed")
    
from triteia.ao.utils.dtypes import BITBLAS_DTYPES
from triteia.ao.utils.bitblas_utils import get_or_create_bitblas_operator

##
## Triton kernel
## 

@autotune.autotune(
    key=["M", "N", "K"],
    nearest_power_of_two=True,
    prune_configs_by={
        "early_config_prune": autotune.matmul248_kernel_config_pruner,
        "perf_model": None,
        "top_k": None,
    },
)
@triton.jit
def quant_matmul_248_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    g_ptr,
    M,
    N,
    K,
    bits,
    maxq,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + (
        (offs_k[:, None] // infearure_per_bits) * stride_bk
        + offs_bn[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(
            scales_ptrs + g_idx[:, None] * stride_scales
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(
            zeros_ptrs + g_idx[:, None] * stride_zeros
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
        b = (b - zeros) * scales  # Scale and shift

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@autotune.autotune(
    key=["M", "N", "K"],
    nearest_power_of_two=True,
)
@triton.jit
def transpose_quant_matmul_248_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    g_ptr,
    M,
    N,
    K,
    bits,
    maxq,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, N) float16
    B is of shape (K//8, N) int32
    C is of shape (M, K) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (
        offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + (
        (offs_bk[:, None] // infearure_per_bits) * stride_bk
        + offs_n[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_bk
    g_idx = tl.load(g_ptrs)

    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = (
        zeros_ptr
        + (offs_n[None, :] // infearure_per_bits)
        + g_idx[:, None] * stride_zeros
    )

    shifter = (offs_bk % infearure_per_bits) * bits
    zeros_shifter = (offs_n % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k in range(0, num_pid_n):
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
        b = (b - zeros) * scales  # Scale and shift
        b = tl.trans(b)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += BLOCK_SIZE_N // infearure_per_bits

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def quant_matmul_248(
    bitwidth, x, qweight, qzero, scale, g_idx, bias: Optional[torch.Tensor] = None
):
    maxq = 2**bitwidth - 1
    with torch.cuda.device(x.device):
        output = torch.empty(
            (x.shape[0], qweight.shape[1]), device=x.device, dtype=x.dtype
        )
        grid = lambda META: (
            triton.cdiv(x.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(qweight.shape[1], META["BLOCK_SIZE_N"]),
        )
        quant_matmul_248_kernel[grid](
            x,
            qweight,
            output,
            scale.to(x.dtype),
            qzero,
            g_idx,
            x.shape[0],
            qweight.shape[1],
            x.shape[1],
            bitwidth,
            maxq,
            x.stride(0),
            x.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            scale.stride(0),
            qzero.stride(0),
        )
        if bias is not None:
            output += bias
        return output


def transpose_quant_matmul_248(input, qweight, scales, qzero, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty(
            (input.shape[0], output_dim), device=input.device, dtype=input.dtype
        )
        grid = lambda META: (
            triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(output_dim, META["BLOCK_SIZE_K"]),
        )
        transpose_quant_matmul_248_kernel[grid](
            input,
            qweight,
            output,
            scales.to(input.dtype),
            qzero,
            g_idx,
            input.shape[0],
            qweight.shape[1],
            output_dim,
            bits,
            maxq,
            input.stride(0),
            input.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            scales.stride(0),
            qzero.stride(0),
        )
        return output


def quant_matmul_inference_only_248(input, qweight, scales, qzero, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty(
            (input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16
        )
        grid = lambda META: (
            triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(qweight.shape[1], META["BLOCK_SIZE_N"]),
        )
        quant_matmul_248_kernel[grid](
            input,
            qweight,
            output,
            scales,
            qzero,
            g_idx,
            input.shape[0],
            qweight.shape[1],
            input.shape[1],
            bits,
            maxq,
            input.stride(0),
            input.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            scales.stride(0),
            qzero.stride(0),
        )
        return output

class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, qweight, scales, qzero, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzero, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzero, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzero, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_quant_matmul_248(
                grad_output, qweight, scales, qzero, g_idx, bits, maxq
            )
        return grad_input, None, None, None, None, None, None


class QuantLinearInferenceOnlyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzero, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzero, g_idx, bits, maxq)
        return output

##
## Bitblas
## 

BITBLAS_PROPAGATE_WEIGHTS = False
BITBLAS_DATABASE_PATH = ".local/bitblas_database"
OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512]

def quant_matmul_248_bitblas(
    bitwidth, x, qweight, qzero, scale, g_idx, bias: Optional[torch.Tensor] = None
):
    pack_factor = Fraction(bitwidth, 8)
    bitblas_dtype = BITBLAS_DTYPES[torch.float16]
    outfeatures = qweight.shape[0]
    infeatures = qweight.shape[1] // pack_factor
    matmul_config = bitblas.MatmulConfig(
            M=x.shape[0],
            N=x.shape[1],
            K=qweight.shape[0],
            A_dtype="float16",
            W_dtype=QUANTIZED_DTYPE[bitwidth],
            accum_dtype="float16",
            out_dtype="float16",
            with_bias=False,
            group_size=qweight.shape[1] * 2,
            with_scaling=True,
            with_zeros=True,
            zeros_mode="quantized",
        )
    bitblas_matmul = get_or_create_bitblas_operator(
        config=matmul_config, enable_tuning=True
    )
    if x.dtype != torch.float16:
        x = x.half()
    output = torch.empty(
        x.shape[:-1] + (qweight.shape[0],),
        dtype=x.dtype, device=x.device
    )
    if bias is not None:
        bitblas_matmul(x, qweight, scale, qzero, output)
    else:
        bitblas_matmul(x, qweight, scale, qzero, bias, output)
    return output

     