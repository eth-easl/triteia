import torch
from triteia.ao.ops import matmul

a = torch.randn((320, 512), device="cuda", dtype=torch.float16)
b = torch.randn((512, 256), device="cuda", dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
