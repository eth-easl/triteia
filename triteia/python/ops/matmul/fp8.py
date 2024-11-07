"""PyTorch Native FP8 Linear Module"""

import torch
import torch.nn as nn

class FP8Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            weight_dtype: torch.dtype,
            bias_dtype: torch.dtype,
            bias: bool=False,
        ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros((out_features, in_features), dtype=weight_dtype))
        self.scale = nn.Parameter(torch.zeros((1), dtype=weight_dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features), dtype=bias_dtype))

    def forward(self, x: torch.Tensor):
        dtype = torch.float8_e4m3fn
        x_f8, x_inv_s = to_float8(x, dtype=dtype)
        y = torch._scaled_mm(
            x_f8, self.weight, out_dtype=torch.float16, scale_a=x_inv_s, scale_b=self.scale
        )

def patch_module_recursive(model: nn.Module):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            patch_module_recursive(module)
        if type(module) == nn.Linear:
            print(f"Patching {n} from {module} to FP8Linear, {type(module)}")
            setattr(model, n, FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    weight_dtype=torch.float8_e4m3fn,
                    bias_dtype=torch.bfloat16,
                    bias=module.bias is not None
                )
            )

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


def fp8_linear(x_fp16, w_data, w_scale, dtype, out_dtype=torch.bfloat16):
