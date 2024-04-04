import torch
import unittest
from ao.ops import bmm
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
        print(self.tensors)


if __name__ == "__main__":
    unittest.main()
