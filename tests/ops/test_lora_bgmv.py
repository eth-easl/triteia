import torch
import unittest
from triteia.python.ops import (
    lora_forloop,
    lora_sgmv,
    lora_bgmv
)
from triteia.python.configs.models.llama import llama_shapes
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_lora_16_bit

# for testing
from triteia.python.capi import add_lora_sgmv_cutlass


class TestLORAOp(unittest.TestCase):
    @torch.inference_mode()
    def run_problem(
        self,
        distribution: str,
        nr: int,
        nm: int,
        m: int,
        n: int,
        rank: int,
        with_base_weight=False,
        dev="cuda",
    ):
        try:
            # nr = number queries (inputs)
            # nm = number of models
            # m = matrix row size
            # n = matrix col size
            # rank = rank
            print(
                f"Running sbmm problem with nr={nr}, nm={nm}, m={m}, n={n}, rank={rank}, distribution={distribution}"
            )
            indices = generate_model_distribution(distribution, nr, nm)
            indices = torch.sort(indices)[0]

            # using n here
            x = torch.randn((nr, n), dtype=torch.float16, device=dev)
            As, Bs = gen_batched_lora_16_bit(
                nm, n, m, rank, device=dev
            )
            native_output = lora_forloop(As, Bs, x, indices, base_weight=None)
            bgvm_output = lora_bgmv(As, Bs, x, indices, base_weight=None)
            
            # Tolerances from punica
            rtol, atol = (5e-3, 5e-3)
            all_close = torch.allclose(native_output, bgvm_output, rtol = rtol, atol = atol)
            print(f"The bgvm_output and native_output {'ARE' if all_close else 'ARE NOT'} close.")

            # Check which individual elements are close
            mask = torch.isclose(native_output, bgvm_output, rtol = rtol, atol = atol)
            num = (~mask).sum().item()
            print(f"Number of elements that are not close: {num}")

            # Print the differences for elements that are not close
            for i in range(min(mask.shape[0], 100)):
                for j in range(min(mask.shape[1], 100)):
                    if not mask[i, j]:
                        diff = native_output[i, j] - bgvm_output[i, j]
                        print(f"Index ({i}, {j}): native_output = {native_output[i, j]}, bgvm_output = {bgvm_output[i, j]}, difference = {diff}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm={nm}, m={m}, n={n}, rank={rank}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        # the rank needs to be divisble by 8
        rank = 64
        self.run_problem("uniform",  10,  5, 768, 768, rank)
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
    print(f'available devices: {torch.cuda.device_count()}')
    print(f'current device: { torch.cuda.current_device()}')
    unittest.main()
