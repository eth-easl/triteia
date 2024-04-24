import bitblas
import torch

in_features = 1024
out_features = 1024
group_size = 128

matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=out_features,  # N dimension
    K=in_features,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="uint4",  # weight W dtype
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
matmul = bitblas.Matmul(config=matmul_config)

# Define shapes for tensors
input_shape = (1, 1024)
weight_shape = (1024, 1024)
scaling_shape = (1024, 1024 // 128)
zeros_shape = (1024, 1024 // 128)
output_shape = (1, 1024)

# Create scaling and zeros tensors for quantization
scaling = torch.rand(scaling_shape, dtype=torch.float16).cuda()
zeros = torch.rand(zeros_shape, dtype=torch.float16).cuda()

# Create input tensor
input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()

# Create and transform weight tensor
weight_tensor = torch.randint(0, 7, weight_shape, dtype=torch.int8).cuda()
weight_tensor_int4 = matmul.transform_weight(weight_tensor)

# Perform mixed-precision matrix multiplication with quantization
output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scaling, zeros=zeros)

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
print("BitBLAS output:", output_tensor)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-2)