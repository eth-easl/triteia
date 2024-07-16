import torch
from triteia.python.ops import (
    sbmm_16bit_forloop,
    sbmm_4bit_2_4_forloop,
    sbmm_4bit_2_4_native,
    sbmm_4bit_2_4_multilaunch,
)
from triteia.python.utils import (
    timing_function,
    print_results_table,
    export_benchmark_results,
)
from triteia.python.configs.models.llama import llama_shapes
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_sparse_quant4_NT

flops_func = lambda nr, m, k: 2 * nr * m * k


def benchmark(distribution, nr, nm, m, k, dev="cuda", groupsize=-1):
    x = torch.randn((nr, k), dtype=torch.float16, device=dev)
    weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
        nr, m, k, groupsize=groupsize, device=dev
    )
    indices = generate_model_distribution(distribution, nr, nm)
    indices = torch.sort(indices)[0]

    def fp16_func(x, weight_ref, indices):
        return sbmm_16bit_forloop(weight_ref, x, indices, base_weight=None)

    def w4_2_4_forloop_func(qweight, x, meta, scale, indices):
        return sbmm_4bit_2_4_forloop(qweight, x, meta, scale, indices, base_weight=None)

    def w4_2_4_native_func(qweight, x, meta, scale, indices):
        return sbmm_4bit_2_4_native(qweight, x, meta, scale, indices, base_weight=None)

    def w4_2_4_multilaunch_func(qweight, x, meta, scale, indices):
        return sbmm_4bit_2_4_multilaunch(
            qweight, x, meta, scale, indices, base_weight=None
        )

    fp16_result = timing_function(
        fp16_func,
        flops_func,
        kwargs={
            "nr": nr,
            "m": m,
            "k": k,
            "x": x,
            "weight_ref": weight_ref,
            "indices": indices,
        },
        repeats=5,
    )
    w4_2_4_forloop_result = timing_function(
        w4_2_4_forloop_func,
        flops_func,
        kwargs={
            "nr": nr,
            "m": m,
            "k": k,
            "qweight": qweight,
            "x": x,
            "meta": meta,
            "scale": scale,
            "indices": indices,
        },
        repeats=5,
    )
    w4_2_4_native_result = timing_function(
        w4_2_4_native_func,
        flops_func,
        kwargs={
            "nr": nr,
            "m": m,
            "k": k,
            "qweight": qweight,
            "x": x,
            "meta": meta,
            "scale": scale,
            "indices": indices,
        },
        repeats=5,
    )
    w4_2_4_multilaunch_result = timing_function(
        w4_2_4_multilaunch_func,
        flops_func,
        kwargs={
            "nr": nr,
            "m": m,
            "k": k,
            "qweight": qweight,
            "x": x,
            "meta": meta,
            "scale": scale,
            "indices": indices,
        },
        repeats=5,
    )
    results = [
        fp16_result,
        w4_2_4_forloop_result,
        w4_2_4_native_result,
        w4_2_4_multilaunch_result,
    ]
    print_results_table(f"sbmm nr={nr},nm={nm},m={m},k={k}", results)
    return results


if __name__ == "__main__":
    results = []
    nr = [16, 32, 64, 128]
    nm = [
        [2, 4, 8, 16],
        [2, 4, 8, 16, 32],
        [2, 4, 8, 16, 32, 64],
        [2, 4, 8, 16, 32, 64, 128],
    ]
    distributions = ["uniform", "zipf:1.5"]
    ms = [4096, 8192]
    ns = [4096, 8192]
    for distribution in distributions:
        for i in range(len(nr)):
            for j in range(len(nm[i])):
                for m in ms:
                    for n in ns:
                        try:
                            results.append(
                                benchmark(distribution, nr[i], nm[i][j], m, n)
                            )
                        except Exception as e:
                            print(
                                f"Failed to benchmark sbmm nr={nr[i]},nm={nm[i][j]},m={m},n={n}"
                            )
                            print(e)
    export_benchmark_results(results, ".local/sbmm_bench.json")
