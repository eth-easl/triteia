import pytest
import torch
import unittest
from triteia.python.ops import (
    bmm_4bit_2_4_forloop,
    gen_batched_sparse_quant4_NT,
    bmm_4bit_2_4,
)
from triteia.python.configs.models.llama import llama_shapes


@pytest.mark.parametrize("b", [4, 8, 16])
@pytest.mark.parametrize("m", [512])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [512])
def test_bmm(b: int, m: int, n: int, k: int, groupsize=-1, dev="cuda"):
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
        diff_forloop = (
            torch.mean(torch.abs(forloop_output - fp16_output)).detach().cpu().numpy()
        )
        diff_native = (
            torch.mean(torch.abs(native_output - fp16_output)).detach().cpu().numpy()
        )
        rel_diff_forloop = (
            diff_forloop / torch.mean(torch.abs(fp16_output)).detach().cpu().numpy()
        )
        rel_diff_native = (
            diff_native / torch.mean(torch.abs(fp16_output)).detach().cpu().numpy()
        )
        # check if all rel_diff_native is less than 1e-3
        assert rel_diff_native < 1e-3
        # check if all rel_diff_forloop is less than 1e-3
        assert rel_diff_forloop < 1e-3
        del x, weight_ref, qweight, scale, meta
    except torch.cuda.OutOfMemoryError as e:
        print(f"Out of memory, skipping b={b} m={m}, n={n}, k={k}")
    finally:
        torch.cuda.empty_cache()
