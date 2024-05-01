import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
import triteia.ao.utils.autotune as autotune
from .matmul_lowprec import quant_matmul_248, quant_matmul_248_bitblas

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
def quant_bmm_248_kernel(
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
    stride_batch_a,
    stride_batch_b,
    stride_batch_c,
    stride_batch_scales,
    stride_batch_zeros,
    stride_batch_g,
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

    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    
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
    a_ptrs = (
        a_ptr
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        + pid_b * stride_batch_a
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = (
        b_ptr
        + (
            (offs_k[:, None] // infearure_per_bits) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        + pid_b * stride_batch_b
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_k + pid_b * stride_batch_g
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn[None, :] + pid_b * stride_batch_scales
    zeros_ptrs = (
        zeros_ptr
        + (offs_bn[None, :] // infearure_per_bits)
        + pid_b * stride_batch_zeros
    )

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

    c_ptrs = (
        c_ptr
        + stride_cm * offs_am[:, None]
        + stride_cn * offs_bn[None, :]
        + pid_b * stride_batch_c
    )

    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, qweight, scales, qzero, g_idx, bits, maxq):
        output = quant_bmm_248(input, qweight, scales, qzero, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzero, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output


def quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx, bias=None):
    maxq = 2**bitwidth - 1
    bsz = x.shape[0]
    output = torch.empty(
        (x.shape[0], x.shape[1], qweight.shape[2]),
        device=x.device,
        dtype=torch.float16,
    )
    grid = lambda META: (
        bsz,
        triton.cdiv(x.shape[1], META["BLOCK_SIZE_M"])
        * triton.cdiv(qweight.shape[2], META["BLOCK_SIZE_N"]),
    )
    src = quant_bmm_248_kernel[grid](
        x,
        qweight,
        output,
        scale,
        qzero,
        g_idx,
        x.shape[1],
        qweight.shape[2],
        x.shape[2],
        bitwidth,
        maxq,
        x.stride(1),
        x.stride(2),
        qweight.stride(1),
        qweight.stride(2),
        output.stride(1),
        output.stride(2),
        scale.stride(1),
        qzero.stride(1),
        x.stride(0),
        qweight.stride(0),
        output.stride(0),
        scale.stride(0),
        qzero.stride(0),
        g_idx.stride(0),
    )
    src = src.asm['ttir']
    with open("quant_bmm_248_kernel.ttir", "w") as f:
        f.write(src)
    if bias is not None:
        output += bias
    return output


class QuantLinearInferenceOnlyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzero, g_idx, bits, maxq):
        output = quant_bmm_248(input, qweight, scales, qzero, g_idx, bits, maxq)
        return output

def loop_quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx, bias=None):
    bsz = x.shape[0]
    output = torch.empty(
        (x.shape[0], x.shape[1], qweight.shape[2]),
        device=x.device,
        dtype=torch.float16,
    )
    for i in range(bsz):
        output[i] = quant_matmul_248(bitwidth, x[i], qweight[i], qzero[i], scale[i], g_idx[i], None)
    if bias is not None:
        output += bias
    return output

def bitblas_loop_quant_bmm_248(bitwidth, x, qweight, qzero, scale, g_idx, bias=None):
    bsz = x.shape[0]
    output = torch.empty(
        (x.shape[0], x.shape[1], qweight.shape[1]),
        device=x.device,
        dtype=torch.float16,
    )
    for i in range(bsz):
        try:
            output[i] = quant_matmul_248_bitblas(
                bitwidth, 
                x[i],
                qweight[i],
                qzero[i],
                scale[i],
                None,
                None
            )
        except Exception:
            print(f"{x[i][0][0]}")
            print(f"{qweight[i][0][0]}")
            print(f"{qzero[i][0][0]}")
            print(f"{scale[i][0][0]}")
            print(f"x[i]: {x[i].shape}@{x[i].device}, qweight[i]: {qweight[i].shape}@{qweight[i].device}, qzero[i]: {qzero[i].shape}@{qzero[i].device}, scale[i]: {scale[i].shape}@{scale[i].device}")
            raise
    if bias is not None:
        output += bias
    return output