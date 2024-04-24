import torch
import unittest
from triteia.ao.ops import bmm, native_bmm_lowprec, quant_bmm_248
from triteia.ao.ops.linalg.matmul.bmm_lowprec import loop_quant_bmm_248
import torch.testing as tt
import safetensors as st

torch.manual_seed(0)


class TestBMM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_bmm(self):
        bszs = [1, 2, 4, 8]
        for bsz in bszs:
            a = torch.randn((bsz, 320, 512), device="cuda", dtype=torch.float16)
            b = torch.randn((bsz, 512, 256), device="cuda", dtype=torch.float16)
            triton_output = bmm(a, b)
            torch_output = torch.bmm(a, b)
            self.assertEqual(triton_output.shape, torch_output.shape)
            tt.assert_close(triton_output, torch_output, rtol=1e-3, atol=3e-5)


class TestBMMLowPrec(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tensors = {}
        with st.safe_open(
            ".local/quantized.safetensors", framework="pt", device="cuda"
        ) as f:
            for key in f.keys():
                self.tensors[key] = f.get_tensor(key)

    def test_lowprec_bmm(self):
        prefix = "model.layers.0.self_attn.q_proj"
        qweight = self.tensors[f"{prefix}.qweight"]
        qzero = self.tensors[f"{prefix}.qzeros"]
        scale = self.tensors[f"{prefix}.scales"]
        g_idx = self.tensors[f"{prefix}.g_idx"]
        bitwidth = 4

        bszs = [1, 2, 4, 8]
        for bsz in bszs:
            x = torch.randn((bsz, 512, 4096), device="cuda", dtype=torch.float16)
            bias = torch.randn((bsz, 512, 4096), device="cuda", dtype=torch.float16)
            qweights = qweight.repeat(bsz, 1, 1)
            qzeros = qzero.repeat(bsz, 1, 1)
            scales = scale.repeat(bsz, 1, 1)
            g_idxs = g_idx.repeat(bsz, 1)
            triton_output = quant_bmm_248(
                bitwidth=bitwidth,
                x=x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            )
            torch_output = native_bmm_lowprec(
                bitwidth=bitwidth,
                x=x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            )
            loop_output = loop_quant_bmm_248(
                bitwidth=bitwidth,
                x=x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=g_idxs,
                bias=bias,
            )
            self.assertEqual(triton_output.shape, torch_output.shape)
            self.assertEqual(triton_output.shape, loop_output.shape)
            tt.assert_close(loop_output, torch_output, rtol=1e-3, atol=3e-5)
            tt.assert_close(triton_output, torch_output, rtol=1e-3, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
