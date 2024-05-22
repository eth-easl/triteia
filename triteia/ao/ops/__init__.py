import torch
from triteia.ao.ops.activations.silu import silu

def default_compile(fn):
    return torch.compile(
        fn,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=True,
    )

__all__ = [
    "silu"
]
