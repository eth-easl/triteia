import torch
import unittest
from triteia.ao.ops import matmul, native_matmul_lowprec_248, quant_matmul_248
from triteia.ao.ops.linalg.matmul.matmul_lowprec import quant_matmul_248_bitblas
import torch.testing as tt
import safetensors as st

torch.manual_seed(0)


# class TestMatmul(unittest.TestCase):
#     def setUp(self):
#         torch.manual_seed(0)

#     def test_matmul(self):
#         a = torch.randn((320, 512), device="cuda", dtype=torch.float16)
#         b = torch.randn((512, 256), device="cuda", dtype=torch.float16)
#         triton_output = matmul(a, b)
#         torch_output = torch.matmul(a, b)
#         self.assertEqual(triton_output.shape, torch_output.shape)
#         tt.assert_close(triton_output, torch_output, rtol=1e-3, atol=3e-5)


class TestMatmulLowPrec(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tensors = {}
        self.bitblas_tensors = {}
        with st.safe_open(
            ".local/quantized.safetensors", framework="pt", device="cuda"
        ) as f:
            for key in f.keys():
                self.tensors[key] = f.get_tensor(key)
        with st.safe_open(
            ".local/bitblas.safetensors", framework="pt", device="cuda"
        ) as f:
            for key in f.keys():
                self.bitblas_tensors[key] = f.get_tensor(key)

    def test_lowprec_matmul(self):
        prefix = "model.layers.0.self_attn.q_proj"
        qweight = self.tensors[f"{prefix}.qweight"]
        qzeros = self.tensors[f"{prefix}.qzeros"]
        scales = self.tensors[f"{prefix}.scales"]
        g_idx = self.tensors[f"{prefix}.g_idx"]

        qweight_bitblas = self.bitblas_tensors[f"{prefix}.qweight"]
        qzeros_bitblas = self.bitblas_tensors[f"{prefix}.zeros"]
        scales_bitblas = self.bitblas_tensors[f"{prefix}.scales"]
        x = torch.rand((256, 4096), device="cuda", dtype=torch.float16)
        bias = torch.rand((256, 4096), device="cuda", dtype=torch.float16)
        # bias = None
        pytorch_output = native_matmul_lowprec_248(
            4, x, qweight, qzeros, scales, g_idx, bias=bias
        )
        triton_output = quant_matmul_248(
            4, x, qweight, qzeros, scales, g_idx, bias=bias
        )
        bitblas_output = quant_matmul_248_bitblas(
            4, x, qweight_bitblas, qzeros_bitblas, scales_bitblas, g_idx, bias=bias)
        print(f"PyTorch output: {pytorch_output}")
        print(f"BitBlas output: {bitblas_output}")
        self.assertEqual(pytorch_output.shape, triton_output.shape)
        self.assertEqual(pytorch_output.shape, bitblas_output.shape)
        # tt.assert_close(pytorch_output, triton_output)
        tt.assert_close(pytorch_output, bitblas_output, rtol=1e-3, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
