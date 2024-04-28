import safetensors as st
import torch
from triteia.ao.nn.linear_bitblas import unpack_qzeros, gptq_unpack_qzeros

def main(args):
    tensors = {}
    with st.safe_open(args.ckpt, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    for key, tensor in tensors.items():
        print(f"{key}: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")
    qzeros = tensors['model.layers.3.self_attn.q_proj.zeros']
    print(qzeros)
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    main(args)
    
    