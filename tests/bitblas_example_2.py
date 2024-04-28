import os
import torch
import bitblas
from triteia.ao.utils.bitblas_utils import get_or_create_bitblas_operator
os.environ["NUMEXPR_MAX_THREADS"] = "32"

in_features = 4096
out_features = 4096
group_size = 4096
bitwidth = 2

def unpack_2bit_to_fp16(qzeros, scale=0.1, zero_point=-2.0):
    # Unpack 2-bit values and interpret them as unsigned integers
    unpacked_zeros = torch.zeros((qzeros.shape[0], qzeros.shape[1] * 16), dtype=torch.float16, device=qzeros.device)
    for col in range(unpacked_zeros.shape[1]):
        i = col % 16
        # Shift to get the 2-bit value, mask it with 0b11 (which is 3 in decimal)
        unpacked_value = (qzeros[:, col // 16] >> (2 * i)) & 0b11
        unpacked_zeros[:, col] = unpacked_value.float()
    return unpacked_zeros

matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=out_features,  # N dimension
    K=in_features,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype=f"uint{bitwidth}",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="original",  # setting for how to calculating zeros
)
# matmul = bitblas.Matmul(config=matmul_config)
matmul = get_or_create_bitblas_operator(config=matmul_config)
# Define shapes for tensors
input_shape = (1, in_features)
weight_shape = (out_features, in_features)
scaling_shape = (in_features, out_features // group_size)
zeros_shape = (in_features, out_features // group_size)
output_shape = (1, out_features)

# Create scaling and zeros tensors for quantization
scaling = torch.rand(scaling_shape, dtype=torch.float16).cuda()
zeros = torch.rand(zeros_shape, dtype=torch.float16).cuda()
# Create input tensor
input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()
# Create and transform weight tensor
maxq = 2 ** (bitwidth - 1) - 1

weight_tensor = torch.randint(0, 4, weight_shape, dtype=torch.int8).cuda()

print(f"weight tensor, min: {weight_tensor.min()}, max: {weight_tensor.max()}")

weight_tensor_quant = matmul.transform_weight(weight_tensor)


print(weight_tensor_quant.shape)
print(f"weight tensor quant, min: {weight_tensor_quant.min()}, max: {weight_tensor_quant.max()}")
print(f"zeros, min: {zeros.min()}, max: {zeros.max()}")
# Perform mixed-precision matrix multiplication with quantization
output_tensor = matmul(input_tensor, weight_tensor_quant, scale=scaling, zeros=zeros)
print("BitBLAS output:", output_tensor)


rescaling_tensor = torch.zeros_like(weight_tensor, dtype=torch.float16).cuda()
# Compute reference result with manual scaling and zero-point adjustment
# rescale = (weight - zeros) * scaling
for i in range(in_features // group_size):
    for j in range(group_size):
        rescaling_tensor[:, i * group_size + j] = (
            weight_tensor[:, i * group_size + j].to(torch.float16) - zeros[:, i]
        ) * scaling[:, i]

ref_result = torch.matmul(input_tensor, rescaling_tensor.t().to(torch.float16))
# Assert that the results are close within a specified tolerance
print("Ref output:", ref_result)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-1, atol=1e-1)
