import torch
import unittest
from triteia.python.ops import to_float8, fp8_mm
import torch.nn.functional as F


class TestFP8(unittest.TestCase):
    def run_problem(self, M, N, K, dtype=torch.float8_e4m3fn):
        x = torch.randn((1, M, N), dtype=torch.float16, device="cuda")
        w = torch.randn((N, K), dtype=torch.float16, device="cuda").t()
        x_f8, x_inv_s = to_float8(x, dtype=dtype)
        w_f8, w_inv_s = to_float8(w)
        y_fp8 = fp8_mm(x, w_f8, out_dtype=torch.float16, scale_b=w_inv_s)
        y_fp16 = torch.matmul(x, w)
        print(f"y_fp8: {y_fp8.shape}, y_fp16: {y_fp16.shape}")
        cos_sim = F.cosine_similarity(y_fp8.reshape(-1), y_fp16.reshape(-1), dim=0)
        return cos_sim

    def test_tiny(self):
        self.run_problem(16, 16, 16, dtype=torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
