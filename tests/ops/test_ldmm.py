import torch
import unittest
from triteia.python.ops import (
    ldmm,
    lora_bgmv,
    sbmm_4bit_2_4_native
)
from triteia.python.configs.models.llama import llama_shapes
from triteia.python.ops.utils.generator import generate_model_distribution
from triteia.python.ops import gen_batched_lora_16_bit
from triteia.python.ops import gen_batched_sparse_quant4_NT

# for testing
from triteia.python.capi import add_lora_sgmv_cutlass


class TestLORAOp(unittest.TestCase):
    @torch.inference_mode()
    def run_problem(
        self,
        distribution: str,
        nr: int,
        nm_lora: int,
        nm_sbmm: int,
        m: int,
        n: int,
        rank: int,
        with_base_weight=False,
        groupsize=-1,
        dev="cuda",
    ):
        try:
            # nr = number queries (inputs)
            # nm = number of models
            # m = matrix row size
            # n = matrix col size
            # rank = rank
            print(
                f"Running sbmm problem with nr={nr}, nm_lora={nm_lora}, nm_sbmm={nm_sbmm}, m={m}, n={n}, rank={rank}, distribution={distribution}"
            )
            nr_lora = nr // 2
            nr_sbmm = nr - nr_lora
            indices_lora = generate_model_distribution(distribution, nr_lora, nm_lora)
            indices_sbmm = generate_model_distribution(distribution, nr_sbmm, nm_sbmm)
            indices_lora = torch.sort(indices_lora)[0]
            indices_sbmm = torch.sort(indices_sbmm)[0]

            # calculate the lora result
            x_lora = torch.randn((nr_lora, n), dtype=torch.float16, device=dev)
            LwA, LwB = gen_batched_lora_16_bit(
                nm_lora, n, m, rank, device=dev
            )
            native_lora_output = lora_bgmv(LwA, LwB, x_lora, indices_lora, base_weight=None)

            # calculate the sbmm result
            x_sbmm = torch.randn((nr_sbmm, n), dtype=torch.float16, device=dev)
            weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
                nm_sbmm, m, n, groupsize=groupsize, device=dev
            )
            native_sbmm_output = sbmm_4bit_2_4_native(
                qweight, x_sbmm, meta, scale, indices_sbmm, base_weight=None
            )

            # combine results
            native_output = torch.cat((native_lora_output, native_sbmm_output), 0)

            # calculate the ldmm result
            x = torch.cat((x_lora, x_sbmm), 0)
            # add number of lora models to the sbmm indices that are not -1
            indices_sbmm[indices_sbmm != -1] += nm_lora
            indices = torch.cat((indices_lora, indices_sbmm), 0)
            ldmm_output = ldmm(indices, x, LwA, LwB, qweight, meta, scale, base_weight=None)
            
            # Tolerances from punica
            rtol, atol = (5e-3, 5e-3)
            all_close = torch.allclose(native_output, ldmm_output, rtol = rtol, atol = atol)
            print(f"The bgvm_output and native_output {'ARE' if all_close else 'ARE NOT'} close.")

            # Check which individual elements are close
            mask = torch.isclose(native_output, ldmm_output, rtol = rtol, atol = atol)
            num = (~mask).sum().item()
            print(f"Number of elements that are not close: {num}")

            # Print the differences for elements that are not close
            for i in range(min(mask.shape[0], 100)):
                for j in range(min(mask.shape[1], 100)):
                    if not mask[i, j]:
                        diff = native_output[i, j] - ldmm_output[i, j]
                        print(f"Index ({i}, {j}): native_output = {native_output[i, j]}, ldmm_output = {ldmm_output[i, j]}, difference = {diff}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm_lora={nm_lora}, nm_sbmm={nm_sbmm}, m={m}, n={n}, rank={rank}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        # the rank needs to be divisble by 8
        rank = 16
        self.run_problem("uniform",  10, 5, 5, 768, 768, rank)
        self.run_problem("zipf:1.5", 128, 2, 2, 4096, 12288, rank)

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
