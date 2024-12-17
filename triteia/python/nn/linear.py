import torch
import torch.nn as nn
import numpy as np
from triteia.python.capi import mul_2_4
from triteia.python.ops.utils.sparsity import (
    _perm_2_4,
    _scale_perm_2_4,
    _scale_perm_single_2_4,
    _perm,
    _scale_perm,
    _scale_perm_single,
    mask_creator,
    sparse_semi_structured_from_dense_cutlass,
)
from triteia.python.ops import fp8_scaled_mm


class sparse_low_precision_linear(nn.Module):
    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """
        Sparse (2:4) and Low Precision Linear Layer
        ----
        Parameters:
            infeatures: `int` number of input features
            outfeatures: `int` number of output features
            groupsize: `int` size of the group
        """
        super().__init__()
        assert groupsize in [-1], f"Group size must be -1, got {groupsize}"
        assert (
            infeatures % 128 == 0
        ), f"Input features must be a multiple of 128, got {infeatures}"
        assert (
            outfeatures % 256 == 0
        ), f"Output features must be a multiple of 256, got {outfeatures}"
        if groupsize == -1:
            groupsize = infeatures
        assert (
            infeatures % groupsize == 0
        ), f"Input features must be a multiple of group size, got {infeatures} and {groupsize}"
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.groupsize = groupsize
        self.qweight = nn.Parameter(torch.empty((self.infeatures // 32, self.outfeatures * 16 // 8), dtype=torch.int32), False)
        self.meta = nn.Parameter(torch.empty((self.outfeatures, self.infeatures // 16), dtype=torch.int16), False)
        self.scales = nn.Parameter(torch.empty((self.infeatures // groupsize, self.outfeatures), dtype=torch.float16), False)
        self.workspace = nn.Parameter(torch.zeros(self.outfeatures // 128 * 16, dtype=torch.int32), False)

    def forward(self, x):
        C = torch.empty(
            x.shape[:-1] + (self.outfeatures,), dtype=x.dtype, device=x.device
        )
        self.workspace = self.workspace.to(x.device)
        mul_2_4(
            x.view((-1, x.shape[-1])),
            self.qweight,
            self.meta,
            C.view((-1, C.shape[-1])),
            self.scales,
            self.workspace,
        )
        return C

    def pack(self, weight, scales, trans=False):
        """
        Pack the weight and scales
        ----
        Parameters:
            weight: `torch.nn.Linear` weight matrix
            scales: `torch.Tensor` scales
        """
        if weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        if trans:
            perm, scale_perm, scale_perm_single = (
                _perm_2_4,
                _scale_perm_2_4,
                _scale_perm_single_2_4,
            )
        else:
            perm, scale_perm, scale_perm_single = _perm, _scale_perm, _scale_perm_single

        tile = 16
        maxq = 2**4 - 1
        s = scales
        w = weight
        if self.groupsize != self.infeatures:
            w = w.reshape((-1, self.groupsize, self.outfeatures))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))

        mask = mask_creator(w.T, n=2, m=4).cuda().bool()
        # mask = torch.ones_like(w.t()).cuda().bool()
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.infeatures:
            w = w.reshape((self.groupsize, -1, self.outfeatures))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.infeatures, self.outfeatures)).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
        w = mask * w.T

        w = w.contiguous()
        w, meta = sparse_semi_structured_from_dense_cutlass(w)
        w = w.t()
        self.infeatures = self.infeatures // 2
        self.groupsize = self.groupsize // 2
        s = s.reshape((-1, self.outfeatures)).contiguous()
        w = w.reshape((self.infeatures // tile, tile, self.outfeatures // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.infeatures // tile, self.outfeatures * tile))
        res = w
        res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i

        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
        self.meta[:, :] = meta.to(self.meta.device)


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
        y = fp8_scaled_mm(x, self.weight.T, self.scale, out_dtype=torch.bfloat16)
        return y
