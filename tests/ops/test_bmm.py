import torch
import unittest
from triteia.python.ops import (
    bmm_4bit_2_4_forloop,
    gen_batched_sparse_quant4_NT,
    bmm_4bit_2_4,
)
from triteia.python.configs.models.llama import llama_shapes


class TestBMMOp(unittest.TestCase):
    def run_problem(self, b: int, m: int, n: int, k: int, groupsize=-1, dev="cuda"):
        try:
            print(f"Running bmm problem with b={b} m={m}, n={n}, k={k}")
            x = torch.randn((b, 1, k), dtype=torch.float16, device=dev)
            weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
                b, m, k, groupsize=groupsize, device=dev
            )
            fp16_output = torch.matmul(x, weight_ref)
            forloop_output = bmm_4bit_2_4_forloop(qweight, x, meta, scale)
            native_output = bmm_4bit_2_4(qweight, x, meta, scale)
            torch.cuda.synchronize()
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
            del x, weight_ref, qweight, scale, meta
        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping b={b} m={m}, n={n}, k={k}")
        finally:

            torch.cuda.empty_cache()

    def test_tiny(self):
        self.run_problem(16, 256, 16, 256, groupsize=-1)
        self.run_problem(16, 512, 16, 512, groupsize=-1)
        self.run_problem(16, 256, 16, 512, groupsize=-1)
        self.run_problem(16, 512, 16, 256, groupsize=-1)
        self.run_problem(8, 256, 16, 256, groupsize=-1)
        self.run_problem(8, 512, 16, 256, groupsize=-1)
        self.run_problem(8, 256, 16, 512, groupsize=-1)
        self.run_problem(8, 512, 16, 256, groupsize=-1)
        self.run_problem(4, 512, 16, 512, groupsize=-1)
        self.run_problem(4, 256, 16, 512, groupsize=-1)
        self.run_problem(4, 256, 16, 512, groupsize=-1)
        self.run_problem(4, 512, 16, 256, groupsize=-1)

    def test_llama(self):
        bszs = [2, 4, 8, 16]
        for _, layers in llama_shapes.items():
            for layer in layers:
                for bsz in bszs:
                    self.run_problem(bsz, layer[1], 16, layer[0])

    def test_llama_uneven(self):
        bszs = [1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        for _, layers in llama_shapes.items():
            for layer in layers:
                for bsz in bszs:
                    self.run_problem(bsz, layer[1], 16, layer[0])


if __name__ == "__main__":
    unittest.main()
