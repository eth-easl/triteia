import os
import torch
import triton
from triteia.ao.ops import native_bmm_lowprec
from triteia.ao.ops.linalg.matmul.bmm_lowprec import quant_bmm_248, loop_quant_bmm_248, bitblas_loop_quant_bmm_248
import safetensors as st

os.environ["NUMEXPR_MAX_THREADS"] = "16"

BITWIDTH = 4
BSZs = [1, 8, 16, 32, 64]
prefix = "model.layers.0.self_attn.q_proj"

tensors = {}
bitblas_tensors = {}
with st.safe_open(".local/quantized.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
qweight = tensors[f"{prefix}.qweight"]
qzero = tensors[f"{prefix}.qzeros"]
scale = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]
with st.safe_open(".local/bitblas.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        bitblas_tensors[key] = f.get_tensor(key)
qweight_bitblas = bitblas_tensors[f"{prefix}.qweight"]
qzero_bitblas = bitblas_tensors[f"{prefix}.zeros"]
scale_bitblas = bitblas_tensors[f"{prefix}.scales"]

def warmup():
    print("Warming up...")
    for bsz in BSZs:
        qweights = qweight.repeat(bsz, 1, 1)
        qweights_bitblas = qweight_bitblas.repeat(bsz, 1, 1)
        
        qzeros = qzero.repeat(bsz, 1, 1)
        qzeros_bitblas = qzero_bitblas.repeat(bsz, 1, 1)
        
        scales = scale.repeat(bsz, 1, 1)
        scales_bitblas = scale_bitblas.repeat(bsz, 1, 1)
        g_idxs = g_idx.repeat(bsz, 1)
        
        x_dim = 4096
        x = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)
        bias = torch.randn((bsz, x_dim, 4096), device="cuda", dtype=torch.float16)
        native_bmm_lowprec(
            BITWIDTH,
            x,
            qweight=qweights,
            qzero=qzeros,
            scale=scales,
            g_idx=g_idxs,
            bias=bias,
        )
        quant_bmm_248(
            BITWIDTH,
            x,
            qweight=qweights,
            qzero=qzeros,
            scale=scales,
            g_idx=g_idxs,
            bias=bias,
        )
        loop_quant_bmm_248(
            BITWIDTH,
            x,
            qweight=qweights,
            qzero=qzeros,
            scale=scales,
            g_idx=g_idxs,
            bias=bias,
        )
        bitblas_loop_quant_bmm_248(
            BITWIDTH,
            x,
            qweight=qweights_bitblas,
            qzero=qzeros_bitblas,
            scale=scales_bitblas,
            g_idx=g_idxs,
            bias=bias,
        )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=BSZs,
        line_arg="provider",
        plot_name="bmm_lowprec",
        line_vals=["torch", "ao", "loop", "bitblas"],
        line_names=["torch", "ao", "loop", "bitblas"],
        args={},
    )
)
def benchmark(B, provider):
    M = 4096
    N = 4096
    K = 4096
    x = torch.randn((B, M, N), device="cuda", dtype=torch.float16)
    bias = torch.randn((B, M, N), device="cuda", dtype=torch.float16)
    quantiles = [0.2, 0.5, 0.75]
    qweights = qweight.repeat(B, 1, 1)
    qweights_bitblas = qweight_bitblas.repeat(B, 1, 1)
    qzeros = qzero.repeat(B, 1, 1)
    qzeros_bitblas = qzero_bitblas.repeat(B, 1, 1)
    scales = scale.repeat(B, 1, 1)
    scales_bitblas = scale_bitblas.repeat(B, 1, 1)
    g_idxs = g_idx.repeat(B, 1)
    
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: native_bmm_lowprec(
                BITWIDTH,
                x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            ),
            quantiles=quantiles,
        )
    if provider == "ao":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: quant_bmm_248(
                BITWIDTH,
                x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            ),
            quantiles=quantiles,
        )
    if provider == "loop":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: loop_quant_bmm_248(
                BITWIDTH,
                x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            ),
            quantiles=quantiles,
        )
    if provider == "bitblas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: bitblas_loop_quant_bmm_248(
                BITWIDTH,
                x,
                qweight=qweights_bitblas,
                qzero=qzeros_bitblas,
                scale=scales_bitblas,
                g_idx=g_idx,
                bias=bias,
            ),
            quantiles=quantiles,
        )
    perf = lambda ms: 2 * B * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


warmup()
benchmark.run(
    print_data=True, show_plots=True, save_path="benchmarks/results/bmm_low_prec"
)
