import torch

BITBLAS_STORAGE_DTYPE = "int8"

TORCH_STORAGE_DTYPE = getattr(torch, BITBLAS_STORAGE_DTYPE)

QUANTIZED_DTYPE = {
    2: "uint2",
    4: "uint4",
    8: "uint8",
}
BITBLAS_DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.half: "float16",
    torch.int8: "int8",
}

DTYPES_BIT = {
    'int8': 8
}
