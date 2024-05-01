import torch
from triteia.ao.utils.gen_tensor import generate_bitblas_weight
from triteia.ao.ops.linalg.matmul.matmul_lowprec import quant_matmul_248_bitblas

# M: 1, N: 4096, K: 2048

M = 1
N = 8192
K = 128
bitwidth = 4

qweight, scales, qzeros = generate_bitblas_weight(bitwidth, K, N, K)

x = torch.randn((N, K), device="cuda:0", dtype=torch.float16)

output = quant_matmul_248_bitblas(
    bitwidth,
    x,
    qweight,
    qzeros,
    scales,
    None,
    None
)