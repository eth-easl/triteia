import torch
from triteia.python.ops import gen_sparse_quant4_NT, matmul_4bit_2_4
from triteia.python.ops.utils.generator import torch_weight_to_sparse_marlin

dev = "cuda"
n = 1
m = 12288
k = 6144
groupsize = -1
tp_size = 8

x = torch.randn((n, k), dtype=torch.float16, device=dev)
weight_ref, qweight, scale, meta = gen_sparse_quant4_NT(
    m, k, groupsize=groupsize, device=dev
)
print(
    f"weight_ref: {weight_ref.shape}, qweight: {qweight.shape}, scale: {scale.shape}, meta: {meta.shape}"
)
fp16_output = torch.matmul(x, weight_ref)
qs_output = matmul_4bit_2_4(qweight, x, meta, scale)

qweights_by_tp, scales_by_tp, metas_by_tp = torch_weight_to_sparse_marlin(
    weight_ref, scale, tp_size=tp_size, chunk_by="column"
)
partial_outputs = []
partial_fp16_outputs = []
for i in range(tp_size):
    tp_weight = weight_ref[:, i * m // tp_size : (i + 1) * m // tp_size].contiguous()
    tp_scales = scale[:, i * m // tp_size : (i + 1) * m // tp_size].contiguous()
    partial_output = matmul_4bit_2_4(
        qweights_by_tp[i], x, metas_by_tp[i], scales_by_tp[i]
    )

    partial_outputs.append(partial_output)
    partial_fp16_output = torch.matmul(x, tp_weight)
    partial_fp16_outputs.append(partial_fp16_output)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
tp_output = torch.cat(partial_outputs, dim=1)
fp16_merged_output = torch.cat(partial_fp16_outputs, dim=1)

print(f"max diff (quant): {torch.max(torch.abs(fp16_output - qs_output))}")
print(
    f"mean diff (tp): {torch.max(torch.abs(tp_output - fp16_output)/torch.mean(torch.abs(fp16_output)))}"
)
print(tp_output - qs_output)
print(fp16_output - fp16_merged_output)
print(
    f"mean diff (fp16): {torch.mean(torch.abs(fp16_output - fp16_merged_output)/torch.mean(torch.abs(fp16_output)))}"
)
# print(f"\n\n{tp_output}\n{qs_output}")
