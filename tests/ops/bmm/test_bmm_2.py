import torch
import time
# perform a 100x4096x4096 bmm in torch
x = torch.randn((100, 1, 4096), device="cuda:0", dtype=torch.float16)
weight = torch.randn((100,4096, 4096), device="cuda:0", dtype=torch.float16)

output = torch.bmm(x, weight)