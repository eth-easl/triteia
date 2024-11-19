import torch
from triteia.python.ops import lora_forloop, lora_bgmv, gen_batched_lora_16_bit

from triteia.python.ops.utils.generator import generate_model_distribution

from triteia.python.utils import (
    timing_function,
    print_results_table,
    export_benchmark_results,
)

# Let A m*rank, B rank*n be the two lora matrices
# We compute y += A*(B*x), where x is the input vector of size n
# and y is the ouptut vector of size y = m
# This is done for nr inputs
flops_func = lambda nr, n, m, rank: nr * (m*(2*rank - 1) + rank*(2*n -1) + m)

def benchmark(distribution, nr, nm, m, n, rank, dev="cuda"):
    indices = generate_model_distribution(distribution, nr, nm)
    indices = torch.sort(indices)[0]
    x = torch.randn((nr, n), dtype=torch.float16, device=dev)
    As, Bs = gen_batched_lora_16_bit(
        nm, n, m, rank, device=dev
    )

    def forloop(As, Bs, x, indices):
        return lora_forloop(As, Bs, x, indices, base_weight=None)

    def bgmv(As, Bs, x, indices):
        return lora_bgmv(As, Bs, x, indices, base_weight=None)

    forloop_result = timing_function(
        forloop,
        flops_func,
        kwargs={
            "dist": distribution,
            "nr": nr,
            "n": n,
            "m": m,
            "rank": rank,
            "As": As,
            "Bs": Bs,
            "x": x,
            "indices": indices
        },
        repeats=5,
    )
    bgmv_result = timing_function(
        bgmv,
        flops_func,
        kwargs={
            "dist": distribution,
            "nr": nr,
            "n": n,
            "m": m,
            "rank": rank,
            "As": As,
            "Bs": Bs,
            "x": x,
            "indices": indices
        },
        repeats=5,
    )
    results = [forloop_result, bgmv_result]
    print_results_table(f"lora nr={nr},nm={nm},m={m}, n={n}, rank={rank}", results)
    return results


if __name__ == "__main__":
    results = []
    nr = [100]
    nm = [
        [2, 4, 8, 16, 32, 64, 100],
    ]
    distributions = ["zipf:2.0"]
    ms = [2048]
    ns = [2048]
    ranks = [32]
    for distribution in distributions:
        for i in range(len(nr)):
            for j in range(len(nm[i])):
                for m in ms:
                    for n in ns:
                        for rank in ranks:
                            try:
                                results.append(
                                    benchmark(distribution, nr[i], nm[i][j], m, n, rank)
                                )
                            except Exception as e:
                                print(
                                    f"Failed to benchmark lora nr={nr[i]},nm={nm[i][j]},m={m},n={n},rank={rank}"
                                )
                                print(e)
    export_benchmark_results(results, ".local/lora_bench.json")