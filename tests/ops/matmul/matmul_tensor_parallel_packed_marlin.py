import torch
from fractions import Fraction
from triteia.utils.io import read_tensors
from triteia.ao.ops.matmul.matmul_lowprec import quant_4_marlin_2_4
import triteia.lib.marlin as marlin

DEVICE = "cuda:0"
BITWIDTH = 4
unpacked_tensors = read_tensors(
    ".local/tinyllama/marlin.tp1.unpacked.safetensors",
    prefix="model.layers.0.self_attn",
    device=DEVICE
)
packed_tensors = read_tensors(
    ".local/tinyllama/marlin.tp1.safetensors",
    prefix="model.layers.0.self_attn",
    device=DEVICE
)
quant_pack_factor = Fraction(BITWIDTH, 32)
print(unpacked_tensors.keys())
q_proj = (unpacked_tensors['q_proj.0.qweight'], unpacked_tensors['q_proj.0.meta'], unpacked_tensors['q_proj.0.scales'])
k_proj = (unpacked_tensors['k_proj.0.qweight'], unpacked_tensors['k_proj.0.meta'], unpacked_tensors['k_proj.0.scales'])
v_proj = (unpacked_tensors['v_proj.0.qweight'], unpacked_tensors['v_proj.0.meta'], unpacked_tensors['v_proj.0.scales'])

qkv_proj = (packed_tensors['qkv_proj.0.qweight'], packed_tensors['qkv_proj.0.meta'], packed_tensors['qkv_proj.0.scales'])

infeatures = q_proj[0].shape[0] * 32
outfeatures = (q_proj[0].shape[1] + k_proj[0].shape[1] + v_proj[0].shape[1]) // 2

print(f"infeatures: {infeatures}, outfeatures: {outfeatures}")

inp = torch.randn((1, infeatures), dtype=torch.float16, device="cuda:0")
qkv_output = torch.zeros((1, outfeatures), dtype=torch.float16, device="cuda:0")
workspace = torch.zeros(qkv_output.shape[1] // 128 * 16, device="cuda:0")
q_output = quant_4_marlin_2_4(inp, q_proj[0], q_proj[1], q_proj[2])
k_output = quant_4_marlin_2_4(inp, k_proj[0], k_proj[1], k_proj[2])
v_output = quant_4_marlin_2_4(inp, v_proj[0], v_proj[1], v_proj[2])

unpacked_qkv_output = torch.cat((q_output, k_output, v_output), dim=1)

marlin.mul_2_4(
    inp,
    qkv_proj[0],
    qkv_proj[1],
    qkv_output,
    qkv_proj[2],
    workspace
)
print(f"unpacked_qkv_output: {unpacked_qkv_output}")
print(f"qkv_output: {qkv_output}")
if not torch.allclose(unpacked_qkv_output, qkv_output, atol=0.01):
    print("Failed")
    print(f"unpacked_qkv_output: {unpacked_qkv_output.shape}")
    print(f"packed_output: {qkv_output.shape}")
    print(f"part 1: {(unpacked_qkv_output[:, :2048] - qkv_output[:, :2048]).max()}")
    print(f"part 2: {(unpacked_qkv_output[:, 2048:2304] - qkv_output[:, 2048:2304]).max()}")
    print(f"part 3: {(unpacked_qkv_output[:, 2304:] - qkv_output[:, 2304:]).max()}")