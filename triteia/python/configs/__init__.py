import torch
from .gpus.specs import *

precisions = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
}
