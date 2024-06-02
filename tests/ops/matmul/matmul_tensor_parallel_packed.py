import torch
from fractions import Fraction
from triteia.utils.io import read_tensors
from triteia.ao.ops.matmul.matmul_lowprec import quant_matmul_248
import triteia.lib.marlin as marlin

DEVICE = "cuda:0"
BITWIDTH = 4
gptq_tensors = read_tensors(
    ".local/tinyllama/gptq.safetensors",
    prefix="model.layers.0.self_attn",
    device=DEVICE
)
marlin_tensors = read_tensors(
    ".local/tinyllama/marlin.tp1.safetensors",
    prefix="model.layers.0.self_attn",
    device=DEVICE
)
quant_pack_factor = Fraction(BITWIDTH, 32)

q_proj = (gptq_tensors['q_proj.qweight'], gptq_tensors['q_proj.qzeros'], gptq_tensors['q_proj.scales'], gptq_tensors['q_proj.g_idx'])
k_proj = (gptq_tensors['k_proj.qweight'], gptq_tensors['k_proj.qzeros'], gptq_tensors['k_proj.scales'], gptq_tensors['k_proj.g_idx'])
v_proj = (gptq_tensors['v_proj.qweight'], gptq_tensors['v_proj.qzeros'], gptq_tensors['v_proj.scales'], gptq_tensors['v_proj.g_idx'])

qkv_proj = (marlin_tensors['qkv_proj.0.qweight'], marlin_tensors['qkv_proj.0.meta'], marlin_tensors['qkv_proj.0.scales'])

infeatures = q_proj[0].shape[0] // quant_pack_factor
outfeatures = q_proj[0].shape[1] + k_proj[0].shape[1] + v_proj[0].shape[1]

print(f"infeatures: {infeatures}, outfeatures: {outfeatures}")

inp = torch.randn((1, infeatures), dtype=torch.float16, device="cuda:0")
qkv_output = torch.zeros((1, outfeatures), dtype=torch.float16, device="cuda:0")
workspace = torch.zeros(qkv_output.shape[1] // 128 * 16, device="cuda:0")


q_output = quant_matmul_248(BITWIDTH, inp, *q_proj, bias=None)
k_output = quant_matmul_248(BITWIDTH, inp, *k_proj, bias=None)
v_output = quant_matmul_248(BITWIDTH, inp, *v_proj, bias=None)

gptq_qkv_output = torch.cat((q_output, k_output, v_output), dim=1)

marlin.mul_2_4(
    inp,
    qkv_proj[0],
    qkv_proj[1],
    qkv_output,
    qkv_proj[2],
    workspace
)
if not torch.allclose(gptq_qkv_output, qkv_output, atol=0.1):
    print("Failed")
    print(f"gptq_qkv_output: {gptq_qkv_output.shape}")
    print(f"qkv_output: {qkv_output.shape}")
    print(f"part 1: {(gptq_qkv_output[:, :2048] - qkv_output[:, :2048]).max()}")
    print(f"part 2: {(gptq_qkv_output[:, 2048:2304] - qkv_output[:, 2048:2304]).max()}")
    print(f"part 3: {(gptq_qkv_output[:, 2304:] - qkv_output[:, 2304:]).max()}")
    
    