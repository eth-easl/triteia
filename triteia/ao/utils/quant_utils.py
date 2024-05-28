# adapted from https://github.com/IST-DASLab/marlin/blob/2e87035acf1b117aaf2c840c32b6a2b0a6c6ca4a/conversion/convert.py
import torch

@torch.no_grad()
def unpack_4bit_to_32bit_signed(qweight, qzeros):
    # Unpack 4-bit values and interpret them as signed integers
    unpacked_weights = torch.zeros(
        (qweight.shape[0]*8, qweight.shape[1]),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False
    )
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1]*8), 
        dtype=torch.int8, 
        device=qzeros.device, 
        requires_grad=False
    )


    for row in range(unpacked_weights.shape[0]):
        i = row % 8
        unpacked_weights[row, :] = (qweight[row // 8, :] >> (4 * i)) & 0xF

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    if not torch.all(unpacked_zeros == 7):
        raise ValueError(
            "Marlin kernel is compatible only with checkpoints using symmetric quantization."
            "Found non-symmetric quantization for the weight"
        )
    return unpacked_weights, unpacked_zeros + 1

@torch.no_grad()
def dequantize_weight(qweight, qzeros, scales):
    unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales
    return unpacked_qweight.T

@torch.no_grad()
def gptq_unpack(bits, qweight, qzeros, scales, group_size=-1):
    if group_size == -1:
        group_size = qweight.shape[0] * 32 // bits
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    wf = wf.to(qweight.device)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int16 if bits == 8 else torch.int8)

    zeros = zeros + 1
    zeros = torch.bitwise_and(
        zeros, (2**bits) - 1
    )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    scales = scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int16 if bits == 8 else torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape(-1, group_size, weight.shape[2])
    weight = scales * (weight - zeros)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight