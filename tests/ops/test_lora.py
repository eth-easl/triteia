import torch
import unittest
from triteia.python.ops import (
    lora_16bit_forloop,
)
from triteia.python.configs.models.llama import llama_shapes
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_lora_16


class TestLORAOp(unittest.TestCase):
    def run_problem(
        self,
        distribution: str,
        nr: int,
        nm: int,
        m: int,
        n: int,
        k: int,
        with_base_weight=False,
        groupsize=-1,
        dev="cuda",
    ):
        try:
            # nr = number queries (inputs)
            # nm = number of models
            # m = matrix row size
            # n = matrix col size
            # k = rank
            print(
                f"Running sbmm problem with nr={nr}, nm={nm}, m={m}, n={n}, k={k}, distribution={distribution}"
            )
            indices = generate_model_distribution(distribution, nr, nm)
            indices = torch.sort(indices)[0]
            # using n here
            x = torch.randn((nr, n), dtype=torch.float16, device=dev)
            # TODO: should I transpose the weight matrice like in the sbmm generator?
            As, Bs = gen_batched_lora_16(
                nm, n, m, k, device=dev
            )
            fp16_output = lora_16bit_forloop(As, Bs, x, indices, base_weight=None)

        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm={nm}, m={m}, n={n}, k={k}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        rank = 50
        self.run_problem("uniform",  10,  5, 256,  256, rank)
        self.run_problem("zipf:1.5", 128, 2, 4096, 12288, rank)

    # def test_llama(self):
    #     nrs = [16, 32, 64, 128, 256]
    #     nms = [[2,4,8,16], [2,4,8,16,32], [2,4,8,16,32,64], [2,4,8,16,32,64,128], [2,4,8,16,32,64,128,256]]
    #     distributions = ["uniform", "zipf:1.5"]
    #     for _, layers in llama_shapes.items():
    #         for layer in layers:
    #             for nr_id, nr in enumerate(nrs):
    #                 for nm_id, nm in enumerate(nms[nr_id]):
    #                     for distribution in distributions:
    #                         self.run_problem(distribution, nr, nm, layer[0], layer[1])


if __name__ == "__main__":
    unittest.main()
