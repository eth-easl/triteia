from tqdm import tqdm
import safetensors as st
import torch, argparse, copy
from triteia.lib.marlin import Layer_2_4 as MarlinLayer
from triteia.lib.marlin.semi_structured_conversions import sparse_semi_structured_from_dense_cutlass
from triteia.utils.io import save_tensors
from triteia.ao.utils.quant_utils import dequantize_weight

@torch.no_grad()
def convert_model(args, verbose=True):
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
        ).to(torch.float16)
        new_module = MarlinLayer(
            infeatures=dequantized_weight.shape[1],
            outfeatures=dequantized_weight.shape[0],
            groupsize=-1
        )
        new_module.pack(
            dequantized_weight,
            scales=copy.deepcopy(tensors[module + ".scales"].t())
        )
        new_tensors[module + ".B"] = new_module.B
        new_tensors[module + ".s"] = new_module.s
        new_tensors[module + ".meta"] = new_module.meta
        
    return new_tensors        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()
    print("Converting model...")
    new_tensors = convert_model(args, verbose=True)
    save_tensors(new_tensors, args.save_path)