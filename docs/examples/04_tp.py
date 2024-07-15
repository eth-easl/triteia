import torch
from triteia.python.ops import gen_sparse_quant4_NT, matmul_4bit_2_4
from triteia.python.utils.quant_utils import unpack_2bit_from_16bit, pack_2bit_to_16bit
dev = "cuda"
n = 1
m = 256
k = 512
groupsize = -1
tp_size = 2

x = torch.randn((n, k), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_sparse_quant4_NT(
    m, k, groupsize=groupsize, device=dev
)
print(f"weight_ref: {weight_ref.shape}, qweight: {qweight.shape}, scale: {scale.shape}, meta: {meta.shape}")
fp16_output = torch.matmul(x, weight_ref)
qs_output = matmul_4bit_2_4(qweight, x, meta, scale)
meta_unpacked = unpack_2bit_from_16bit(meta.cpu())
meta_unpacked = meta_unpacked.reshape((k, m//2))
print(meta_unpacked.shape)
partial_outputs = []
for i in range(tp_size):
    tp_qweight = qweight[:, i * k // tp_size: (i + 1) * k // tp_size]
    tp_scale = scale[:, i * m // tp_size: (i + 1) * m // tp_size]
    tp_meta = meta_unpacked[:, i * m //2// tp_size: (i + 1) * m //2// tp_size]
    print(f"tp_meta: {tp_meta.shape}")
    tp_meta = pack_2bit_to_16bit(tp_meta).cuda().reshape((128, 32))
    print(f"tp_qweight: {tp_qweight.shape}, tp_scale: {tp_scale.shape}, tp_meta: {tp_meta.shape}")
    partial_output = matmul_4bit_2_4(tp_qweight, x, tp_meta, tp_scale)
    partial_outputs.append(partial_output)
    
tp_output = torch.cat(partial_outputs, dim=1)
print(f"max diff (quant): {torch.max(torch.abs(fp16_output - qs_output))}")
print(f"max diff (tp): {torch.max(torch.abs(tp_output - qs_output))}")

# torch.cuda.synchronize()
