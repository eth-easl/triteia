import torch
import unittest
from ao.ops import matmul
import torch.testing as tt

torch.manual_seed(0)

class TestMatmul(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matmul(self):
        a = torch.randn((320, 512), device='cuda', dtype=torch.float16)
        b = torch.randn((512, 256), device='cuda', dtype=torch.float16)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a, b)
        self.assertEqual(triton_output.shape, torch_output.shape)
        tt.assert_close(triton_output, torch_output, rtol=1e-3, atol=3e-5)

if __name__ == '__main__':
    unittest.main()