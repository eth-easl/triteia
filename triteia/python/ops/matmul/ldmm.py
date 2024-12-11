import torch
from triteia.python.capi import add_lora_bgmv
from triteia.python.capi import sbmm_2_4
from triteia.python.capi import mul_2_4
from triteia.python.ops import sbmm_4bit_2_4_forloop, lora_forloop

def baseline_ldmm(indices, x, LwA, LwB, DeltaW, metas, ss, base_weight=None):

    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    else:
        y = torch.zeros(x.shape[0], LwB.shape[2], dtype=x.dtype, device=x.device)
    if torch.all(indices == -1):
        return y

    M = LwA.shape[0]
    N = DeltaW.shape[0]

    mask_lora = (indices < M) & (indices != -1)
    mask_sbmm = (indices >= M) & (indices != -1)

    # ================== lora calculation ===============================
    if (mask_lora.sum() > 0):
        x_lora = x[mask_lora]
        y_lora = y[mask_lora]
        indices_lora = indices[mask_lora]
        indices_lora = indices_lora.to(torch.long)

        y_lora = lora_forloop(LwA, LwB, x_lora, indices_lora)
        y[mask_lora] = y_lora

     # ================== sbmm calculation ===============================
    if (mask_sbmm.sum() > 0):
        x_sbmm = x[mask_sbmm]
        y_sbmm = y[mask_sbmm]
        indices_sbmm = indices[mask_sbmm] - M
        y_sbmm = sbmm_4bit_2_4_forloop(DeltaW, x_sbmm, metas, ss, indices_sbmm)
        y[mask_sbmm] = y_sbmm

    return y




def ldmm (indices, x, LwA, LwB, DeltaW, metas, ss, base_weight=None):
    """
    Args:
        indices: Shape: `[B]`. `B` is the batch size. This tensor contains the index of the model to use for each
            query vector in the batch. \
            The indices can be in range `[0:M+N-1]`, where `M` is the number of lora models and `N` is the \
            number of sbmm models. \
            An index `i < M` represents the `i`'th lora model,\
            whereas an index `i >= M` represents the `(i-M)`'th sbmm model.
        x: Shape: `[B, H1]`. A batch containing `B` query vectors of size H1.
        LwA: Shape: `[M, H1, R]`. M weight matrices of size `H1 x R`, `R` being the rank of the lora models
        LwB: Shape: `[M, R, H2]`. M weight matrices of size `R x H2`, `R` being the rank of the lora models
        DeltaW: `N` quantized weight matrices for the sbmm models
        metas: `N` times meta data for the sbmm models
        ss: `N` time scale data for the sbmm models
    Returns:
        y: Shape: `[B, H2]`, where each row corresponds to the output for a query vector in `x` \
            and the selcted model .

    """

    if base_weight is not None:
        y = torch.matmul(x, base_weight.t())
    else:
        y = torch.zeros(x.shape[0], LwB.shape[2], dtype=x.dtype, device=x.device)
    if torch.all(indices == -1):
        return y

    M = LwA.shape[0]
    N = DeltaW.shape[0]

    mask_lora = (indices < M) & (indices != -1)
    mask_sbmm = (indices >= M) & (indices != -1)
    # ================== lora calculation ===============================
    if (mask_lora.sum() > 0):

        x_lora = x[mask_lora]
        y_lora = y[mask_lora]
        indices_lora = indices[mask_lora]
        indices_lora = indices_lora.to(torch.long)

        # Transpose and ensure contiguity after transpose
        LwA_T = LwA.transpose(1, 2).contiguous()
        LwB_T = LwB.transpose(1, 2).contiguous()
    
        # Add layer dimension
        LwA_T = LwA_T.unsqueeze(1)
        LwB_T = LwB_T.unsqueeze(1)

        add_lora_bgmv(y_lora, x_lora, LwA_T, LwB_T, indices_lora, layer_idx = 0, scale = 1.0)

        y[mask_lora] = y_lora

   
    # ================== sbmm calculation ===============================
    if (mask_sbmm.sum() > 0):
        x_sbmm = x[mask_sbmm]
        y_sbmm = y[mask_sbmm]
        indices_sbmm = indices[mask_sbmm] - M
        unique_sbmm_indices, counts = torch.unique_consecutive(indices_sbmm, return_counts=True)
        if len(unique_sbmm_indices) == 1:
            # use a normal matmul
            workspace = torch.zeros(
                y_sbmm.shape[1] // 128 * 16, device=x_sbmm.device, dtype=torch.int32
            )
            output = torch.zeros_like(y_sbmm)
            mul_2_4(
                x_sbmm,
                DeltaW[unique_sbmm_indices[0]],
                metas[unique_sbmm_indices[0]],
                output,
                ss[unique_sbmm_indices[0]],
                workspace,
            )
            y_sbmm += output
        else:
            unique_sbmm_indices = unique_sbmm_indices.int()
            counts = counts.int()
            first_nonnegative = torch.where(indices_sbmm != -1)[0][0]
            assert(first_nonnegative == 0) # since the sbmm indices do not include the -1's this should always hold
            if first_nonnegative > 0:
                unique_sbmm_indices = unique_sbmm_indices[1:]
                counts = counts[1:]
            start = torch.cat(
                (
                    torch.tensor([first_nonnegative]).cuda(),
                    (torch.cumsum(counts, dim=0) + first_nonnegative)[:-1],
                )
            ).int()
            workspace = torch.zeros(
                len(unique_sbmm_indices), y_sbmm.shape[1] // 8, device=x_sbmm.device, dtype=torch.int32
            )
            output = torch.zeros(
                (x_sbmm.shape[0], y_sbmm.shape[1]), dtype=torch.float16, device=x_sbmm.device
            )
            sbmm_2_4(
                x_sbmm,
                DeltaW,
                metas,
                output,
                ss,
                unique_sbmm_indices,
                workspace,
                start,
                counts,
            )
            y_sbmm += output
        
        y[mask_sbmm] = y_sbmm

    return y