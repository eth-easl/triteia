import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from triteia.python.utils.gpu import is_hopper, is_ampere
from triteia.python.utils import warn_once

available_impl = ["torch"]

try:
    if is_hopper():
        from flash_attn_interface import flash_attn_func

        available_impl.append("fa")
    elif is_ampere():
        from flash_attn import flash_attn_func

        available_impl.append("fa")
    else:
        warn_once("flash_attn is not supported on this GPU, using torch instead")
except ImportError:
    warn_once("flash_attn is not installed, using torch instead")


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
    impl: str = "fa",
):
    # qkv should be of the same shape
    assert q.shape == k.shape == v.shape
    if attn_mask is not None:
        impl = "torch"
    if impl not in available_impl:
        impl = "torch"
    # qkv are of shape (batch, seq_len, num_heads, head_dim)
    if impl == "torch":
        output = F.scaled_dot_product_attention(
            torch.permute(q, [0, 2, 1, 3]),
            torch.permute(k, [0, 2, 1, 3]),
            torch.permute(v, [0, 2, 1, 3]),
            attn_mask=attn_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=scale,
        )
        output = torch.permute(output, [0, 2, 1, 3])
    elif impl == "fa":
        output = flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal)

    return output
