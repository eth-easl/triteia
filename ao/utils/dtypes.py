import torch

BITBLAS_STORAGE_DTYPE = "int8"
TORCH_STORAGE_DTYPE = getattr(torch, BITBLAS_STORAGE_DTYPE)
QUANTIZED_DTYPE = {
    2: "uint2",
    4: "uint4",
    8: "uint8",
}