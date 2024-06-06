import torch
import triteia.lib.marlin as marlin

def bmm_sparse_marlin_forloop(bitwidth, x, qweight, scale, metas):
    outputs = torch.zeros(
        (x.shape[0], scale.shape[2]), dtype=x.dtype, device=x.device
    )
    # if all indices are -1, return y
    for id in range(x.shape[0]):
        workspace = torch.zeros(x.shape[1] // 128 * 16, device=x.device)
        output = torch.zeros((
            1, x.shape[1]
        ), dtype=torch.float16, device=x.device)
        marlin.mul_2_4(
            x,
            qweight[id],
            metas[id],
            output,
            scale[id],
            workspace,
        )
        outputs[id] = output
    return outputs

def bmm_native(bitwidth, x, qweight, scale, metas):
    outputs = torch.zeros(
        (x.shape[0], scale.shape[2]), dtype=x.dtype, device=x.device
    )
    workspace = torch.zeros(16384 // 128 * 16, device=x.device)
    marlin.bmm_2_4(
        x,
        qweight,
        metas,
        outputs,
        scale,
        workspace
    )
    return outputs