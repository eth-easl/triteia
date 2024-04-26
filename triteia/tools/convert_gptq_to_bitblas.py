import os
import safetensors as st
from safetensors.torch import save_file
from tqdm import tqdm
from triteia.ao.utils.bitblas_utils import convert_to_bitblas

def main(args):
    print(args)
    tensors = {}
    new_tensors = {}
    remaining_keys = []
    with st.safe_open(args.ckpt, framework="pt") as f:
        remaining_keys = list(f.keys())
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    for module in tqdm(quantized_modules):
        qweight, scales, zeros, bias = convert_to_bitblas(
            args.bitwidth, module, tensors, args.zeros_mode
        )
        new_tensors[module + ".qweight"] = qweight
        new_tensors[module + ".scales"] = scales
        new_tensors[module + ".zeros"] = zeros
        if bias is not None:
            new_tensors[module + ".bias"] = bias
        remaining_keys.remove(module + ".qweight")
        remaining_keys.remove(module + ".qzeros")
        remaining_keys.remove(module + ".scales")
        remaining_keys.remove(module + ".g_idx")
    new_tensors.update({key: tensors[key] for key in remaining_keys})        
    
    print(f"Finished converting to bitblas with bitwidth {args.bitwidth}! Saving to {args.output}...")
    
    save_file(new_tensors, args.output)

if __name__ == "__main__":
    import os
    import argparse
    os.environ["NUMEXPR_MAX_THREADS"] = "16"    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bitwidth", type=int, required=True)
    parser.add_argument("--zeros-mode", type=str, default="quantized")
    args = parser.parse_args()
    main(args)
