import torch
import unittest
from triteia.ao.ops import bmm, native_bmm_lowprec, quant_bmm_248
from triteia.ao.ops.linalg.matmul.bmm_lowprec import loop_quant_bmm_248, bitblas_loop_quant_bmm_248
import torch.testing as tt
import safetensors as st

torch.manual_seed(0)

device = "cuda:1"

class TestBMMLowPrec(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tensors = {}
        with st.safe_open(
            ".local/4bit_bitblas.safetensors", framework="pt", device=device
        ) as f:
            for key in f.keys():
                self.tensors[key] = f.get_tensor(key)

    def test_lowprec_bmm(self):
        prefix = "model.layers.0.self_attn.q_proj"
        self.tensors = {
            key: value.to(device)
            for key, value in self.tensors.items()
        }
        qweight = self.tensors[f"{prefix}.qweight"]
        qzero = self.tensors[f"{prefix}.zeros"]
        scale = self.tensors[f"{prefix}.scales"]
        print(qweight.device)
        bitwidth = 4
        bszs = [1, 2, 4, 8]
        
        for bsz in bszs:
            x = torch.randn((bsz, 512, 4096), device=device, dtype=torch.float16)
            bias = torch.randn((bsz, 512, 4096), device=device, dtype=torch.float16)
            qweights = qweight.repeat(bsz, 1, 1)
            qzeros = qzero.repeat(bsz, 1, 1)
            scales = scale.repeat(bsz, 1, 1)
            bitblas_output = bitblas_loop_quant_bmm_248(
                bitwidth=bitwidth,
                x=x,
                qweight=qweights,
                qzero=qzeros,
                scale=scales,
                g_idx=None,
                bias=bias,
            )
            print(bitblas_output)
if __name__ == "__main__":
    unittest.main()
