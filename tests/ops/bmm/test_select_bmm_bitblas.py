import torch
import safetensors as st
from triteia.ao.ops.linalg.select_matmul.select_bmm import bitblas_quant_select_bmm_248
from triteia.ao.ops.linalg.select_matmul.select_bmm import ibmm

if __name__=="__main__":
    MAX_DELTAS = 4
    TOKEN_LENGTH = 16
    HIDDEN_DIM = 4096
    triton_weight = ".local/4bit_bitblas.safetensors"
    tensors = {}
    prefix = "model.layers.0.self_attn.q_proj"
    with st.safe_open(triton_weight, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    qweight = tensors[f"{prefix}.qweight"]
    qzeros = tensors[f"{prefix}.zeros"]
    scales = tensors[f"{prefix}.scales"]
    
    qweights = qweight.repeat(MAX_DELTAS, 1, 1)
    qzeros = qzeros.repeat(MAX_DELTAS, 1, 1)
    scales = scales.repeat(MAX_DELTAS, 1, 1)
    
    x = torch.rand((TOKEN_LENGTH, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    y = torch.zeros((TOKEN_LENGTH, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    
    indices = torch.randint(-1, MAX_DELTAS, (TOKEN_LENGTH,), device="cuda")
    print(f"indices: {indices}")
    
    my_y = ibmm(4, indices,y, x, qweights, qzeros, scales, None)
    
    print(f"my y: {my_y}")
    
    y = torch.zeros((TOKEN_LENGTH, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    bitblas_quant_select_bmm_248(4, indices, y, x, qweights, qzeros, scales, None)
    naive_y = y.clone()
    print(f"naive y: {naive_y}")
    print(f"allclose: {torch.allclose(my_y, naive_y, atol=1e-3, rtol=1e-3)}")