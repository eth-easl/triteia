import triton
import torch
from ao.ops import native_bmm_lowprec, quant_bmm_248
import safetensors as st

tensors = {}
with st.safe_open(".local/quantized.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

BITWIDTH = 4
prefix = "model.layers.0.self_attn.q_proj"
qweight = tensors[f"{prefix}.qweight"]
qzero = tensors[f"{prefix}.qzeros"]
scale = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]


def warmup():
    print("Warming up...")
    for bsz in [1,2,4,8]:
        qweights = qweight.repeat(bsz, 1, 1)
        qzeros = qzero.repeat(bsz, 1,1)
        scales = scale.repeat(bsz, 1,1)
        g_idxs = g_idx.repeat(bsz, 1)
        for i in range(1, 15):
            x_dim = int(128 * i)
            
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[1,2,4,8],
        line_arg="provider",
        plot_name="bmm_lowprec",
        line_vals=["torch", "ao"],
        line_names=["torch", "ao"],
        args={},
    )
)
def benchmark(B, provider):
    M = 512
    N = 4096
    K = 4096
    x = torch.randn((B, M, N), device="cuda", dtype=torch.float16)
    bias = torch.randn((B, M, N), device="cuda", dtype=torch.float16)
    quantiles = [0.2, 0.5, 0.75]
    qweights = qweight.repeat(B, 1, 1)
    qzeros = qzero.repeat(B, 1,1)
    scales = scale.repeat(B, 1,1)
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
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

warmup()
benchmark.run(
    print_data=True, show_plots=True, save_path="benchmarks/results/bmm_low_prec"
)
