import torch
import unittest
from triteia.python.ops import (
    lora_forloop,
    lora_sgmv
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
            sgvm_output = lora_sgmv(As, Bs, x, indices, base_weight=None)
            
            """
                reproducing their testing
            """
            # batch_setup = "3x7"
            # num_problems = nm
            # problem_size = nr/nm
            # dtype = torch.float16
            # num_layers = 1
            # device = dev

            # wa = [
            #     torch.rand((num_layers, n, rank), dtype=dtype, device=device)
            #     for _ in range(num_problems)
            # ]
            # wb = [
            #     torch.rand((num_layers, rank, m), dtype=dtype, device=device)
            #     for _ in range(num_problems)
            # ]
            # wa_ptr = torch.tensor([t.data_ptr() for t in wa], dtype=torch.int64, device=device)
            # wb_ptr = torch.tensor([t.data_ptr() for t in wb], dtype=torch.int64, device=device)
            # s = torch.cumsum(
            #     torch.tensor([0] + [problem_size] * num_problems, device=device),
            #     dim=0,
            #     dtype=torch.int32,
            # )
            
            # x = torch.rand((s[-1], n), dtype=dtype, device=device)
            # y = torch.rand((s[-1], m), dtype=dtype, device=device)

            # add_lora_sgmv_cutlass(y, x, wa_ptr, wb_ptr, s, 0, rank)
            
            print(torch.allclose(native_output, sgvm_output))

        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm={nm}, m={m}, n={n}, rank={rank}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        rank = 50
        self.run_problem("uniform",  10,  5, 256,  256, rank)
        #self.run_problem("zipf:1.5", 128, 2, 4096, 12288, rank)

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
