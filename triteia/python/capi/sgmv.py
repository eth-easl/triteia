import torch
import triteia_cuda

def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, R]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    tmp_size = triteia_cuda.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    triteia_cuda.sgmv_cutlass(v, x, wa_ptr, s, tmp, layer_idx)
    triteia_cuda.sgmv_cutlass(y, v, wb_ptr, s, tmp, layer_idx)

def add_lora_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ wa_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          @ wb_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      wa_T_all: Shape: `[None, L, R, H1]`. All of the transposed LoRA A matrices.
      wb_T_all: Shape: `[None, L, H2, R]`. All of the transposed LoRA B matrices.
      indicies: Shape: `[B]`. Indices of the LoRA weights.
      layer_idx: Layer index of LoRA weights.
      scale: Scaling factor.
    """
    f = triteia_cuda.dispatch_bgmv
    device = x.device
    dtype = x.dtype

    r = wb_T_all.size(-1)
    tmp = torch.zeros((x.size(0), r), dtype=dtype, device=device)
    f(tmp, x, wa_T_all, indicies, layer_idx, 1.0)
    f(y, tmp, wb_T_all, indicies, layer_idx, scale)
