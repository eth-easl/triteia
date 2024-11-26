"""PyTorch Native FP8 Linear Module"""

import torch
import torch.nn as nn
from triteia.python.utils import vprint
from triteia.python.configs import precisions


def fp8_scaled_mm(
    x: torch.Tensor, w_f8: torch.Tensor, scale_b: torch.Tensor, out_dtype=torch.float16
):
    needs_reshape = False

    if x.dim() == 3:
        needs_reshape = True
        if x.size(0) != 1:
            raise ValueError("Expected input to have a batch size of 1")
        x = x.squeeze(0)

    x_f8, x_inv_s = to_float8(x, dtype=w_f8.dtype)
    y = torch._scaled_mm(
        x_f8,
        w_f8,
        out_dtype=out_dtype,
        scale_a=x_inv_s,
        scale_b=scale_b,
    )
    if needs_reshape:
        y = y.unsqueeze(0)
    return y


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def mixed_precision_load(model, tensors, strict: bool = False):
    model.load_state_dict(tensors, strict=strict)
    return model
