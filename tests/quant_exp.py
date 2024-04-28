import torch
from triteia.ao.nn.linear_bitblas import unpack_qzeros, gptq_pack_zeros
import numpy as np
in_features = 1024
out_features = 1024
group_size = out_features
bitwidth = 2
zeros_shape = (in_features, out_features // group_size)

def unpack_2bit_to_fp16(qzeros, bitwidth):
    # Unpack 2-bit values and interpret them as unsigned integers
    unpacked_zeros = torch.zeros((qzeros.shape[0], qzeros.shape[1] * 16), dtype=torch.float16, device=qzeros.device)
    for col in range(unpacked_zeros.shape[1]):
        i = col % 16
        # Shift to get the 2-bit value, mask it with 0b11 (which is 3 in decimal)
        unpacked_value = (qzeros[:, col // 16] >> (2 * i)) & 0b11
        unpacked_zeros[:, col] = unpacked_value.float()
    return unpacked_zeros

zeros = torch.rand(zeros_shape, dtype=torch.float16).cuda()
gptq_zeros = zeros.T

print(f"zeros: {gptq_zeros}")
# pack
gptq_zeros = gptq_zeros.cpu()
intzeros = gptq_zeros.numpy().astype(np.uint32)
print(intzeros)

packed_qzeros = gptq_pack_zeros(gptq_zeros, bitwidth)
print(f"packed_zeros, {packed_qzeros}")

# unpack
intzeros = unpack_2bit_to_fp16(packed_qzeros, bitwidth).T.contiguous()
print(f"unpacked_zeros, {intzeros}")

