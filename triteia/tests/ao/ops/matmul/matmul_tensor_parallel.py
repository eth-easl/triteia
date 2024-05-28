import torch
import triteia.lib.marlin as marlin
from triteia.utils.generator import generate_2_4_pruned

m = 2048
k = 5632

fp16, qs, scales, metas = generate_2_4_pruned(1, m, k)
inp = torch.randn((1, k), dtype=torch.float16, device="cuda:0")
ref_output = torch.zeros((1, m), dtype=torch.float16, device="cuda:0")

workspace = torch.zeros(k // 128 * 16, device=torch.device('cuda:0'))
qweight = qs[0]
scales = scales[0]
meta = metas[0]
fp16 = fp16[0]
marlin.mul_2_4(
    inp,
    qweight,
    meta,
    ref_output,
    scales,
    workspace,
)
fp16_ref_output = torch.matmul(inp, fp16)
# tensor parallel - column
out_features = qweight.shape[1] // 2

qs_a = qweight[:, :out_features]
qs_b = qweight[:, out_features:]
meta_a = meta[:out_features//2, :]
meta_b = meta[out_features//2:, :]
scales_a = scales[:, :out_features//2]
scales_b = scales[:, out_features//2:]
f16_weight_a = fp16[:, :out_features]
f16_weight_b = fp16[:, out_features:]

output_a = torch.zeros((1, m//2), dtype=torch.float16, device="cuda:0")
output_b = torch.zeros((1, m//2), dtype=torch.float16, device="cuda:0")

marlin.mul_2_4(
    inp,
    qs_a,
    meta_a,
    output_a,
    scales_a,
    workspace,
)
marlin.mul_2_4(
    inp,
    qs_b,
    meta_b,
    output_b,
    scales_b,
    workspace,
)
fp16_output_a = torch.matmul(inp, f16_weight_a)
fp16_output_b = torch.matmul(inp, f16_weight_b)

output = torch.cat([output_a, output_b], dim=1)
fp16_output = torch.cat([fp16_output_a, fp16_output_b], dim=1)

print(f"reference fp16: {fp16_ref_output}")
print(f"tensor parallel fp16: {fp16_output}")
print("-"*20)
print(f"reference: {ref_output}")
print(f"tensor parallel: {output}")