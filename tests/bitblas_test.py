import bitblas
import torch

matmul_config = bitblas.MatmulConfig(
    M=2048,  # M dimension
    N=4096,  # N dimension
    K=11008,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="int4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)

matmul = bitblas.Matmul(config=matmul_config)

# Create input matrices
input_tensor = torch.rand((2048, 4096), dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, 7, (11008, 2048), dtype=torch.int8).cuda()

# Transform weight tensor to int4 data type
# Perform mixed-precision matrix multiplication
output_tensor = matmul(input_tensor, weight_tensor)

# Reference result using PyTorch matmul for comparison
ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))
# Assert that the results are close within a specified tolerance, note that the int4 randint value is a little bigger than the float16 value, so we set the atol to 1.0
print("Ref output:", ref_result)
print("BitBLAS output:", output_tensor)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-0)
