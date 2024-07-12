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
    def run_problem_column(
        self,
        distribution: str,
        nr: int,
        nm: int,
        m: int,
        k: int,
        tp_size: int,
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
            ref_weights, qweights, scales, metas = [], [], [], []
            ref_fp16_outputs = []
            outputs = []
            for i in range(tp_size):
                weight_ref, qweight, scale, meta = gen_batched_sparse_quant4_NT(
                    nm, m, k, groupsize=groupsize, device=dev
                )
                ref_weights.append(weight_ref)
                qweights.append(qweight)
                scales.append(scale)
                metas.append(meta)
                
                fp16_partial_output = sbmm_16bit_forloop(weight_ref, x, indices, base_weight=None)
                native_partial_output = sbmm_4bit_2_4_native(
                    qweight, x, meta, scale, indices, base_weight=None
                )
                ref_fp16_outputs.append(fp16_partial_output)
                outputs.append(native_partial_output)
            
            ref_fp16_final_outputs = torch.cat(ref_fp16_outputs, dim=1)
            final_outputs = torch.cat(outputs, dim=1)
            
            stacked_fp16_weights = torch.cat(ref_weights, dim=2)
            stacked_qweights = torch.cat(qweights, dim=2)
            stacked_scales = torch.cat(scales, dim=2)
            stacked_metas = torch.cat(metas, dim=1)
            
            stacked_fp16_output = sbmm_16bit_forloop(stacked_fp16_weights, x, indices, base_weight=None)
            stacked_native_output = sbmm_4bit_2_4_native(
                stacked_qweights, x, stacked_metas, stacked_scales, indices, base_weight=None
            )
            self.assertLess(
                torch.mean(torch.abs(final_outputs - ref_fp16_final_outputs))
                / torch.mean(torch.abs(ref_fp16_final_outputs)),
                0.002,
            )
            self.assertLess(
                torch.mean(torch.abs(stacked_fp16_output - ref_fp16_final_outputs))
                / torch.mean(torch.abs(ref_fp16_final_outputs)),
                0.002,
            )
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"Out of memory, skipping nr={nr}, nm={nm}, m={m}, k={k}")
        finally:
            torch.cuda.empty_cache()

    def test_tiny(self):
        self.run_problem_column("uniform",  10,  5, 256,  256, 2)
        

if __name__ == "__main__":
    unittest.main()
