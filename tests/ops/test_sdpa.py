import torch
import unittest
from triteia.python.ops import sdpa


class TestSDPAOp(unittest.TestCase):
    def run_problem(self, q, k, v, is_causal, impl):
        return sdpa(q, k, v, is_causal=is_causal, impl=impl)

    def test_tiny(self):
        bsz = 1
        seq_len = 16
        num_heads = 4
        head_dim = 4
        q = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
        k = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
        v = torch.randn((bsz, seq_len, num_heads, head_dim), dtype=torch.float16).cuda()
        output = self.run_problem(q, k, v, is_causal=True, impl="fa")
        ref = self.run_problem(q, k, v, is_causal=True, impl="torch")
        print(torch.max(torch.abs(output - ref)))
        self.assertTrue(torch.allclose(output, ref, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
