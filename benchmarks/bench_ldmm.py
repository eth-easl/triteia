import torch
from triteia.python.ops import ldmm, lora_bgmv, sbmm_4bit_2_4_native
from triteia.python.ops import gen_batched_lora_16_bit, gen_batched_sparse_quant4_NT

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
# We also add the number of flops for the nr_sbmm inputs
flops_func = lambda nr_sbmm, nr_lora, n, m, rank: nr_lora * (m*(2*rank - 1) + rank*(2*n -1) + m) + 2 * nr_sbmm * m * n

def benchmark(distribution, nr_lora, nr_sbmm, nm_lora, nm_sbmm, m, n, rank, groupsize=-1, dev="cuda"):
    indices_lora = generate_model_distribution(distribution, nr_lora, nm_lora)
    indices_sbmm = generate_model_distribution(distribution, nr_sbmm, nm_sbmm)
    indices_lora = torch.sort(indices_lora)[0]
    indices_sbmm = torch.sort(indices_sbmm)[0]
    x_lora = torch.randn((nr_lora, n), dtype=torch.float16, device=dev)
    x_sbmm = torch.randn((nr_sbmm, n), dtype=torch.float16, device=dev)

    x_ldmm = torch.cat((x_lora, x_sbmm), 0)
    # add number of lora models to the sbmm indices that are not -1
    indices_sbmm[indices_sbmm != -1] += nm_lora
    indices_ldmm = torch.cat((indices_lora, indices_sbmm), 0)

    As, Bs = gen_batched_lora_16_bit(
        nm_lora, n, m, rank, device=dev
    )

    weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
        nm_sbmm, m, n, groupsize=groupsize, device=dev
    )

    def native(As, Bs, qweight, scale, meta, x_lora, x_sbmm, indices_lora, indices_sbmm):
        native_sbmm_output = sbmm_4bit_2_4_native(
            qweight, x_sbmm, meta, scale, indices_sbmm, base_weight=None
        )
        native_lora_output = lora_bgmv(As, Bs, x_lora, indices_lora, base_weight=None)
        return torch.cat((native_lora_output, native_sbmm_output), 0)

    def ldmm_bench(As, Bs, qweight, scale, meta, x, indices):
        return ldmm(indices, x, As, Bs, qweight, meta, scale, base_weight=None)
    
    # native_result = timing_function(
    #     native,
    #     flops_func,
    #     kwargs={
    #         "dist": distribution,
    #         "nr_lora": nr_lora,
    #         "nr_sbmm": nr_sbmm,
    #         "n": n,
    #         "m": m,
    #         "rank": rank,
    #         "As": As,
    #         "Bs": Bs,
    #         "qweight": qweight,
    #         "scale": scale,
    #         "meta": meta,
    #         "x_lora": x_lora,
    #         "x_sbmm": x_sbmm,
    #         "indices_lora": indices_lora,
    #         "indices_sbmm": indices_sbmm
    #     },
    #     repeats=5,
    # )
    ldmm_result = timing_function(
        ldmm_bench,
        flops_func,
        kwargs={
            "dist": distribution,
            "nr_lora": nr_lora,
            "nr_sbmm": nr_sbmm,
            "n": n,
            "m": m,
            "rank": rank,
            "As": As,
            "Bs": Bs,
            "qweight": qweight,
            "scale": scale,
            "meta": meta,
            "x": x_ldmm,
            "indices": indices_ldmm
        },
        repeats=5,
    )
    results = [ldmm_result]
    print_results_table(f"lora nr={nr_lora},nm={nm_lora},m={m}, n={n}, rank={rank}", results)
    return results


if __name__ == "__main__":
    results = []
    nr = [100]
    nm = [
        [1, 1, 2, 4, 8, 16, 32, 64, 100],
    ]
    distributions = ["zipf:2.0"]
    ms = [4096]
    ns = [4096, 8192]
    ranks = [32]
    for rank in ranks:
        for distribution in distributions:
            for i in range(len(nr)):
                for j in range(len(nm[i])):
                    for m in ms:
                        for n in ns:
                            # same number of lora and sbmm models
                            try:
                                results.append(
                                    benchmark(distribution, nr[i], nr[i], nm[i][j], nm[i][j], m, n, rank)
                                )
                            except Exception as e:
                                print(
                                    f"Failed to benchmark lora nr={nr[i]},nm={nm[i][j]},m={m},n={n},rank={rank}"
                                )
                                print(e)
    export_benchmark_results(results, ".local/ldmm_bench.json")