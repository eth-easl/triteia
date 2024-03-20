import torch
import unittest
from ao.ops import bmm
import torch.testing as tt

torch.manual_seed(0)

class TestBMM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_bmm(self):
        bszs = [1, 2, 4, 8]
        for bsz in bszs:
            a = torch.randn((bsz, 320, 512), device='cuda', dtype=torch.float16)
            b = torch.randn((bsz, 512, 256), device='cuda', dtype=torch.float16)
            triton_output = bmm(a, b)
            torch_output = torch.bmm(a, b)
            self.assertEqual(triton_output.shape, torch_output.shape)
            tt.assert_close(triton_output, torch_output, rtol=1e-3, atol=3e-5)

if __name__ == '__main__':
    unittest.main()