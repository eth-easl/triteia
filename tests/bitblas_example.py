import bitblas
import torch
import os
os.environ['NUMEXPR_MAX_THREADS'] = "32"

M = 1
N = 1024
K = 1024

GROUP_SIZE = 128
matmul_config = bitblas.MatmulConfig(
    M=M,  # M dimension
    N=N,  # N dimension
    K=K,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="uint2",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=128,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="quantized",  # setting for how to calculating zeros
)

matmul = bitblas.Matmul(config=matmul_config)
scaling_shape = (1024, 1024//128)
zeros_shape = (1024, 1024//128)

# Create input matrices
input_tensor = torch.rand((1, K), dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, 4, (N, K), dtype=torch.int8).cuda()

scaling = torch.rand(scaling_shape, dtype=torch.float16).cuda()
zeros = torch.rand(zeros_shape, dtype=torch.float16).cuda()

# Transform weight tensor to int4 data type
transformed = matmul.transform_weight(weight_tensor, zeros=zeros)
weight_tensor_transformed = transformed[0]
zeros_transformed = transformed[1]
# Perform mixed-precision matrix multiplication
output_tensor = matmul(input_tensor, weight_tensor_transformed, scale=scaling, zeros=zeros_transformed)

# Reference result using PyTorch matmul for comparison
ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))
# Assert that the results are close within a specified tolerance, note that the int4 randint value is a little bigger than the float16 value, so we set the atol to 1.0
print("Ref output:", ref_result)
print("BitBLAS output:", output_tensor)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-0)