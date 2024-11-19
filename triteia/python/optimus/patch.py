import torch
from torch import nn
from triteia.python.nn.linear import FP8Linear
from triteia.python.utils import vprint


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
