import torch
import safetensors as st
from triteia.ao.ops.ibmm.ibmm_marlin import ibmm_sparse_marlin
from triteia.utils.generator import generate_2_4_pruned
from triteia.ao.utils.distribution import generate_model_distribution

DEV = "cuda:0"

if __name__=="__main__":
    k = 5632
    m = 2048
    num_requests = 1
    num_models = 1
    distribution = "uniform"
    indices = generate_model_distribution(distribution, num_requests, num_models)
    indices = torch.sort(indices)[0]
    indices = torch.tensor([0])
    fp16, qs, scales, metas = generate_2_4_pruned(num_models, m, k)
    groupsize = -1
    workspace = torch.zeros(m // 128 * 16, device=DEV, dtype=torch.int32)
    x = torch.randn((num_requests, k), dtype=torch.float16, device=DEV)
    ref_output = torch.zeros((num_requests, m), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin(
        4,indices, metas, ref_output, x, qs, scales
    )
    # now split the scales, meta and qs, say by column now
    infeatures = k
    out_features = qs.shape[2] // 2 # n
    print(f"qs.shape: {qs.shape}, meta: {metas.shape}, scales: {scales.shape}")
    
    qs_a = qs[:, :, :out_features]
    qs_b = qs[:, :, out_features:]
    meta_a = metas[:, :out_features//2, :]
    meta_b = metas[:, out_features//2:, :]
    scales_a = scales[:, :, :out_features//2]
    scales_b = scales[:, :, out_features//2:]
    assert not torch.allclose(qs_a, qs_b)
    assert not torch.allclose(meta_a, meta_b)
    assert not torch.allclose(scales_a, scales_b)
    
    print(f"qs_a.shape: {qs_a.shape}, meta_a: {meta_a.shape}, scales_a: {scales_a.shape}")
    output_a = torch.zeros((num_requests, m//2), dtype=torch.float16, device=DEV)
    output_b = torch.zeros((num_requests, m//2), dtype=torch.float16, device=DEV)
    ibmm_sparse_marlin(
        4, indices, meta_a, output_a, x, qs_a, scales_a
    )
    ibmm_sparse_marlin(
        4, indices, meta_b, output_b, x, qs_b, scales_b
    )
    print(f"reference: {ref_output}")
    # now concatenate the outputs
    print(output_a)
    print(output_b)
    output = torch.cat([output_a, output_b], dim=1)
    print(f"tensor parallel: {output}")
    