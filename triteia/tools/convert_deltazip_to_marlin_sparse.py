from tqdm import tqdm
import safetensors as st
import torch, argparse, copy
from triteia.lib.marlin import Layer_2_4 as MarlinLayer
from triteia.lib.marlin.semi_structured_conversions import sparse_semi_structured_from_dense_cutlass
from triteia.utils.io import save_tensors
from triteia.ao.utils.quant_utils import dequantize_weight

@torch.no_grad()
def convert_model(args, verbose=True):
    DEV = "cuda:0"
    tensors = {}
    new_tensors = {}
    with st.safe_open(args.ckpt, framework="torch", device="cuda:0") as f:
        keys = f.keys()
        for key in keys:
            tensors[key] = f.get_tensor(key)
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    pbar = tqdm(quantized_modules, position=0, leave=True)
    for module in pbar:
        pbar.set_description(f"{module}")
        dequantized_weight = dequantize_weight(
            tensors[module + ".qweight"],
            tensors[module + ".qzeros"],
            tensors[module + ".scales"],
        ).to(torch.float16).t()
        scales = tensors[module + ".scales"]
        # k=5632, m = 2048
        k, m = dequantized_weight.shape[0], dequantized_weight.shape[1]
        k_sp = k // 2
        layer = MarlinLayer(
            infeatures=dequantized_weight.shape[1],
            outfeatures=dequantized_weight.shape[0],
            groupsize=-1
        )
        layer.n = m
        layer.k = k
        layer.groupsize = k
        layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int, device=DEV)
        # 2048, 352
        layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
        layer.s = torch.empty((k_sp // (k // 2), m), dtype=torch.half, device=DEV)
        layer.pack(
            dequantized_weight,
            scales=scales,
            trans=True,
        )
        new_tensors[module + ".B"] = layer.B
        new_tensors[module + ".s"] = layer.s
        new_tensors[module + ".meta"] = layer.meta
    return new_tensors        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()
    print("Converting model...")
    new_tensors = convert_model(args, verbose=True)
    save_tensors(new_tensors, args.save_path)