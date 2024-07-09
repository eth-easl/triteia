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
        n: int,
        k: int,
        with_base_weight=False,
        groupsize=-1,
        dev="cuda",
    ):
        try:
            print(
                f"Running bmm problem with nr={nr}, nm={nm}, m={m}, n={n}, k={k}, distribution={distribution}"
            )
            indices = generate_model_distribution(distribution, nr, nm)
            indices = torch.sort(indices)[0]
            x = torch.randn((nr, k), dtype=torch.float16, device=dev)
            weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
                nr, m, k, groupsize=groupsize, device=dev
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
            self.assertTrue(torch.allclose(forloop_output, native_output, atol=1e-3))
            self.assertTrue(
                torch.allclose(forloop_output, multilaunch_output, atol=1e-3)
            )

        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm={nm}, m={m}, n={n}, k={k}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        self.run_problem("uniform", 10, 5, 256, 16, 256, groupsize=-1)


if __name__ == "__main__":
    unittest.main()
