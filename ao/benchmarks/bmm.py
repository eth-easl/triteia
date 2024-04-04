import triton
import torch
from ao.ops import bmm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[i for i in range(2, 15)],
        line_arg="provider",
        plot_name="batched_matmul",
        line_vals=["torch", "ao"],
        line_names=["torch", "ao"],
        args={},
    )
)
def benchmark(B, provider):
    M = 512
    K = 512
    N = 512
    a = torch.randn((B, M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((B, K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.2, 0.5, 0.75]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.bmm(a, b), quantiles=quantiles
        )
    if provider == "ao":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: bmm(a, b), quantiles=quantiles
        )
    perf = lambda ms: 2 * B * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(
    print_data=True, show_plots=True, save_path="ao/benchmarks/results/square_matmul"
)
