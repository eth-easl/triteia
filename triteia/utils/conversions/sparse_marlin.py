import torch
from triteia.lib.marlin import Layer_2_4 as MarlinLayer

@torch.no_grad()
def torch_weight_to_sparse_marlin(weight, scale, tp_size=1, chunk_by="column"):
    """
    Args:
        weight: torch.Tensor of shape (in_features, out_features)
        scale: torch.Tensor of shape (1, out_features)
        tp_size: tensor parallelism size
        chunk_by: "column" or "row"
    """
    assert chunk_by in ["column", "row"], "chunk_by must be either 'column' or 'row'"
    assert weight.dim() == 2, "weight must be a 2D tensor"
    assert weight.size(0) % tp_size == 0, "out_features must be divisible by tp_size"
    if not weight.is_contiguous():
        weight = weight.contiguous()
    for i in range(tp_size):
        if chunk_by == "column":
            tp_weight = weight[
                :, 
                i * weight.size(1) // tp_size: (i + 1) * weight.size(1) // tp_size
            ]
            tp_scales = scale[
                :, 
                i * weight.size(1) // tp_size: (i + 1) * weight.size(1) // tp_size
            ]
        else:
            tp_weight = weight[
                i * weight.size(0) // tp_size: (i + 1) * weight.size(0) // tp_size, 
                :
            ]
            tp_scales = scale
        layer = MarlinLayer(
            infeatures=tp_weight.size(0),
            outfeatures=tp_weight.size(1),
            groupsize=-1
        )
        k, m = tp_weight.size(0), tp_weight.size(1)
        k_sp = k // 2
        layer.groupsize = k
        layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int)
        layer.meta = torch.empty((m, k // 16), dtype=torch.int16)
        layer.s = torch.empty((k_sp // (k // 2), m), dtype=torch.half)
        layer.pack(tp_weight, scales=tp_scales, trans=True)
        return layer.B, layer.s, layer.meta