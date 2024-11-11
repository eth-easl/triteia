"""PyTorch Native FP8 Linear Module"""

import torch
import torch.nn as nn
from triteia.python.utils import vprint
from triteia.python.configs import precisions


def fp8_mm(
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


class FP8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        bias: bool = False,
    ) -> None:
        """Forward-only Linear layer with FP8 weights and BFloat16 bias"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), dtype=weight_dtype),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros((out_features), dtype=bias_dtype),
                requires_grad=False,
            )
        self.dtype = torch.float8_e4m3fn

    def forward(self, x: torch.Tensor):
        # if x has 3 dimensions, squeeze the first dimension
        y = fp8_mm(x, self.weight.T, self.scale, out_dtype=torch.bfloat16)
        return y


def patch_module_recursively(
    model: nn.Module, module_name="", ignore_keys: list = [], verbose: bool = True
):
    for n, module in model.named_children():
        new_module_name = f"{module_name}.{n}"
        if len(list(module.children())) > 0:
            patch_module_recursively(
                module, new_module_name, ignore_keys=ignore_keys, verbose=verbose
            )

        if any([k in new_module_name for k in ignore_keys]):
            pass

        elif type(module) == nn.Linear and n not in ignore_keys:
            vprint(
                f"Patching {module_name} from {type(module)} to FP8Linear, {type(module)}",
                verbose,
            )
            setattr(
                model,
                n,
                FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    weight_dtype=torch.float8_e4m3fn,
                    bias_dtype=torch.bfloat16,
                    bias=module.bias is not None,
                ),
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


def mixed_precision_load(model, tensors, strict: bool = False):
    model.load_state_dict(tensors, strict=strict)
    return model
