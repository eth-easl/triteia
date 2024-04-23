import torch
import safetensors as st
from triteia.ao.ops.linalg.select_matmul.select_bmm import quant_select_bmm_248
from triteia.ao.ops.linalg.select_matmul.naive_select_bmm import naive_quant_select_bmm_248

if __name__=="__main__":
    MAX_DELTAS = 4
    TOKEN_LENGTH = 16
    HIDDEN_DIM = 4096
    triton_weight = ".local/quantized.safetensors"
    tensors = {}
    prefix = "model.layers.0.self_attn.q_proj"
    with st.safe_open(triton_weight, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    qweight = tensors[f"{prefix}.qweight"]
    qzeros = tensors[f"{prefix}.qzeros"]
    scales = tensors[f"{prefix}.scales"]
    g_idx = tensors[f"{prefix}.g_idx"]
    
    qweights = qweight.repeat(MAX_DELTAS, 1, 1)
    qzeros = qzeros.repeat(MAX_DELTAS, 1, 1)
    scales = scales.repeat(MAX_DELTAS, 1, 1)
    g_idx = g_idx.repeat(MAX_DELTAS, 1)
    
    x = torch.rand((TOKEN_LENGTH, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    y = torch.zeros((TOKEN_LENGTH, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    
    indices = torch.randint(-1, MAX_DELTAS, (TOKEN_LENGTH,), device="cuda")
    print(f"indices: {indices}")
    
    my_y = quant_select_bmm_248(4, indices,y, x, qweights, qzeros, scales, g_idx)
    
    print(f"my y: {my_y}")
    
    y = torch.zeros((TOKEN_LENGTH, HIDDEN_DIM), device="cuda", dtype=torch.float16)
    naive_quant_select_bmm_248(4, indices, y, x, qweights, qzeros, scales, g_idx)
    naive_y = y.clone()
    print(f"naive y: {naive_y}")
    print(f"allclose: {torch.allclose(my_y, naive_y)}")