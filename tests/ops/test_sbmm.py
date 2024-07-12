import torch
import unittest
from triteia.python.ops import (
    sbmm_16bit_forloop,
    sbmm_4bit_2_4_forloop,
    sbmm_4bit_2_4_native,
    sbmm_4bit_2_4_multilaunch,
)
from triteia.python.configs.models.llama import llama_shapes
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_sparse_quant4_NT


class TestSBMMOp(unittest.TestCase):
    def run_problem(
        self,
        distribution: str,
        nr: int,
        nm: int,
        m: int,
        k: int,
        with_base_weight=False,
        groupsize=-1,
        dev="cuda",
    ):
        try:
            print(
                f"Running sbmm problem with nr={nr}, nm={nm}, m={m}, k={k}, distribution={distribution}"
            )
            indices = generate_model_distribution(distribution, nr, nm)
            indices = torch.sort(indices)[0]
            x = torch.randn((nr, k), dtype=torch.float16, device=dev)
            weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
                nm, m, k, groupsize=groupsize, device=dev
            )
            fp16_output = sbmm_16bit_forloop(weight_ref, x, indices, base_weight=None)
            forloop_output = sbmm_4bit_2_4_forloop(
                qweight, x, meta, scale, indices, base_weight=None
            )
            multilaunch_output = sbmm_4bit_2_4_multilaunch(
                qweight, x, meta, scale, indices, base_weight=None
            )
            native_output = sbmm_4bit_2_4_native(
                qweight, x, meta, scale, indices, base_weight=None
            )

            self.assertLess(
                torch.mean(torch.abs(forloop_output - fp16_output))
                / torch.mean(torch.abs(fp16_output)),
                0.002,
            )
            self.assertLess(
                torch.mean(torch.abs(native_output - fp16_output))
                / torch.mean(torch.abs(fp16_output)),
                0.002,
            )
            self.assertLess(
                torch.mean(torch.abs(multilaunch_output - fp16_output))
                / torch.mean(torch.abs(fp16_output)),
                0.002,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm={nm}, m={m}, k={k}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        self.run_problem("uniform",  10,  5, 256,  256)
        self.run_problem("zipf:1.5", 128, 2, 4096, 12288)
        
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
