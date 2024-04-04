import triton
import torch
from ao.ops import quant_matmul_248, native_matmul_lowprec_248
import safetensors as st

tensors = {}
with st.safe_open(".local/quantized.safetensors", framework="pt", device="cuda") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

BITWIDTH = 4
prefix = "model.layers.0.self_attn.q_proj"
qweight = tensors[f"{prefix}.qweight"]
qzero = tensors[f"{prefix}.qzeros"]
scales = tensors[f"{prefix}.scales"]
g_idx = tensors[f"{prefix}.g_idx"]


def warmup():
    print("Warming up...")
    for i in range(1, 15):
        x_dim = int(128 * i)
        x = torch.randn((x_dim, 4096), device="cuda", dtype=torch.float16)
        bias = torch.randn((x_dim, 4096), device="cuda", dtype=torch.float16)
        native_matmul_lowprec_248(
            BITWIDTH,
            x,
            qweight=qweight,
            qzero=qzero,
            scale=scales,
            g_idx=g_idx,
            bias=bias,
        )
        quant_matmul_248(
            BITWIDTH,
            x,
            qweight=qweight,
            qzero=qzero,
            scale=scales,
            g_idx=g_idx,
            bias=bias,
        )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[128 * i for i in range(2, 15)],
        line_arg="provider",
        plot_name="matmul_lowprec",
        line_vals=["torch", "ao"],
        line_names=["torch", "ao"],
        args={},
    )
)
def benchmark(M, provider):
    N = 4096
    K = 4096
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    bias = torch.randn((M, N), device="cuda", dtype=torch.float16)
    quantiles = [0.2, 0.5, 0.75]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: native_matmul_lowprec_248(
                BITWIDTH,
                x,
                qweight=qweight,
                qzero=qzero,
                scale=scales,
                g_idx=g_idx,
                bias=bias,
            ),
            quantiles=quantiles,
        )
    if provider == "ao":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: quant_matmul_248(
                BITWIDTH,
                x,
                qweight=qweight,
                qzero=qzero,
                scale=scales,
                g_idx=g_idx,
                bias=bias,
            ),
            quantiles=quantiles,
        )
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


warmup()
benchmark.run(
    print_data=True, show_plots=True, save_path="benchmarks/results/matmul_low_prec"
)
