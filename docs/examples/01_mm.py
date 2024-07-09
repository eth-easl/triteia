import torch
from triteia.python.ops import gen_sparse_quant4_NT, matmul_4bit_2_4

dev = "cuda"
n = 1
m = 256
k = 512
groupsize = -1

# x (1, 512)
# weight_ref (512, 256)
# qweight (16, 512) --> (512, 256)
x = torch.randn((n, k), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_sparse_quant4_NT(
    m, k, groupsize=groupsize, device=dev
)
# weight_ref = weight_ref.permute(0, 2, 1)
fp16_output = torch.matmul(x, weight_ref)
qs_output = matmul_4bit_2_4(qweight, x, meta, scale)
print(
    f"weight_ref: {weight_ref.shape}, qweight: {qweight.shape}, scale: {scale.shape}, meta: {meta.shape}"
)
print(fp16_output)
print(qs_output)
torch.cuda.synchronize()
