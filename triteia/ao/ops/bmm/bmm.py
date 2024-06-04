import torch
import triteia.lib.marlin as marlin

def bmm_sparse_marlin_forloop(bitwidth,metas, x, qweight, scale):
    # if all indices are -1, return y
    print(qweight.shape)
    
    # for id, count in zip(unique_indices, counts):
    #     if id != -1:
    #         workspace = torch.zeros(y.shape[1] // 128 * 16, device=x.device)
    #         idx_mask = indices == id
    #         inp = x[idx_mask]
    #         output = torch.zeros((
    #             inp.shape[0], y.shape[1]
    #         ), dtype=torch.float16, device=x.device)
    #         marlin.mul_2_4(
    #             inp,
    #             qweight[id],
    #             metas[id],
    #             output,
    #             scale[id],
    #             workspace,
    #         )
    #         y[idx_mask] += output
    # return y