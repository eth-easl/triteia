import torch
from triteia.python.ops import (
    bmm_4bit_2_4_forloop,
    gen_batched_sparse_quant4_NT,
    bmm_4bit_2_4,
)
from triteia.python.utils import timing_function, print_results_table
from triteia.python.configs.models.llama import llama_shapes

flops_func = lambda b, m, n, k: 2 * b * m * n * k


def benchmark(b, m, n, k, dev="cuda", groupsize=-1):
    x = torch.randn((b, 1, k), dtype=torch.float16, device=dev)
    weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
        b, m, k, groupsize=groupsize, device=dev
    )

    def fp16_func(x, weight_ref):
        return torch.bmm(x, weight_ref)

    def w4_2_4_forloop(qweight, x, meta, scale):
        return bmm_4bit_2_4_forloop(qweight, x, meta, scale)

    def w4_2_4_native(qweight, x, meta, scale):
        return bmm_4bit_2_4(qweight, x, meta, scale)

    w4_2_4_forloop_result = timing_function(
        w4_2_4_forloop,
        flops_func,
        kwargs={
            "b": b,
            "m": m,
            "n": n,
            "k": k,
            "qweight": qweight,
            "x": x,
            "meta": meta,
            "scale": scale,
        },
        repeats=5,
    )
    fp16_result = timing_function(
        fp16_func,
        flops_func,
        kwargs={"b": b, "m": m, "n": n, "k": k, "x": x, "weight_ref": weight_ref},
        repeats=5,
    )
    w4_2_4_native_result = timing_function(
        w4_2_4_native,
        flops_func,
        kwargs={
            "b": b,
            "m": m,
            "n": n,
            "k": k,
            "qweight": qweight,
            "x": x,
            "meta": meta,
            "scale": scale,
        },
        repeats=5,
    )
    results = [fp16_result, w4_2_4_forloop_result, w4_2_4_native_result]
    print_results_table(f"bmm b={b},m={m},n={n},k={k}", results)


if __name__ == "__main__":
    benchmark(2, 256, 32, 256)
    benchmark(8, 4096, 32, 4096)
    benchmark(8, 8192, 8, 8192)
