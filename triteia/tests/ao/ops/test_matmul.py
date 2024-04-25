import os
import torch
import unittest
from triteia.ao.ops import matmul, native_matmul_lowprec_248, quant_matmul_248
from triteia.ao.ops.linalg.matmul.matmul_lowprec import quant_matmul_248_bitblas
import torch.testing as tt
import safetensors as st

torch.manual_seed(0)
os.environ["NUMEXPR_MAX_THREADS"]="16"

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
        self.prefix = "model.layers.0.self_attn.q_proj"
        self.bitwidth = [2, 4]
        self.tensors = {
            'bitblas': {k: {} for k in self.bitwidth},
            'gptq': {k: {} for k in self.bitwidth},
        }
        for bitwidth in self.bitwidth:
            with st.safe_open(
                f".local/{bitwidth}bit_bitblas.safetensors", framework="pt", device="cuda"
            ) as fp:
                for key in fp.keys():
                    self.tensors['bitblas'][bitwidth][key] = fp.get_tensor(key)
            with st.safe_open(
                f".local/{bitwidth}bit_gptq.safetensors", framework="pt", device="cuda"
            ) as fp:
                for key in fp.keys():
                    self.tensors['gptq'][bitwidth][key] = fp.get_tensor(key)

    def test_lowprec_matmul(self):
        for bitwidth in self.bitwidth:
            prefix = "model.layers.0.self_attn.q_proj"
            qweight = self.tensors['gptq'][bitwidth][f"{prefix}.qweight"]
            qzeros = self.tensors['gptq'][bitwidth][f"{prefix}.qzeros"]
            scales = self.tensors['gptq'][bitwidth][f"{prefix}.scales"]
            g_idx = self.tensors['gptq'][bitwidth][f"{prefix}.g_idx"]

            qweight_bitblas = self.tensors['bitblas'][bitwidth][f"{prefix}.qweight"]
            qzeros_bitblas = self.tensors['bitblas'][bitwidth][f"{prefix}.zeros"].T
            scales_bitblas = self.tensors['bitblas'][bitwidth][f"{prefix}.scales"]
            
            x = torch.rand((1, 4096), device="cuda", dtype=torch.float16)
            bias = torch.zeros((1, 4096), device="cuda", dtype=torch.float16)
            pytorch_output = native_matmul_lowprec_248(
                bitwidth, x, qweight, qzeros, scales, g_idx, bias=bias
            )
            triton_output = quant_matmul_248(
                bitwidth, x, qweight, qzeros, scales, g_idx, bias=bias
            )
            bitblas_output = quant_matmul_248_bitblas(
                bitwidth, x, qweight_bitblas, qzeros_bitblas, scales_bitblas, g_idx, bias=bias
            )
            print(f"PyTorch output: {pytorch_output}")
            print(f"BitBlas output: {bitblas_output}")
            self.assertEqual(pytorch_output.shape, triton_output.shape)
            self.assertEqual(pytorch_output.shape, bitblas_output.shape)
            tt.assert_close(pytorch_output, triton_output)
            tt.assert_close(pytorch_output, bitblas_output, rtol=1e-2, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
