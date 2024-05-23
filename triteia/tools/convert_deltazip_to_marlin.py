import os
import copy
import torch
from tqdm import tqdm
import safetensors as st
from safetensors.torch import save_file
from triteia.lib.marlin import Layer as MarlinLayer
from triteia.ao.utils.quant_utils import dequantize_weight

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
        qweight = tensors[module + '.qweight'].to("cuda:0")
        qzeros  = tensors[module + '.qzeros'].to("cuda:0")
        scales  = tensors[module + '.scales'].to("cuda:0")
        dequantized_weight = dequantize_weight(qweight, qzeros, scales).to(torch.float16)
        
        linear_module = torch.nn.Linear(
            in_features=dequantized_weight.shape[1],
            out_features=dequantized_weight.shape[0],
            bias=False,
            dtype=torch.float16,
            device="cuda"
        )
        linear_module.weight.data.copy_(dequantized_weight)
        new_module = MarlinLayer(
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            groupsize=-1
        )
        new_module.pack(
            dequantized_weight, scales=copy.deepcopy(scales.T)
        )
        new_tensors[module + '.qweight'] = new_module.B
        new_tensors[module + '.scales'] =  new_module.s
    save_file(new_tensors, args.output)
    print("All Done!")
    
if __name__ == "__main__":
    import os
    import argparse
    os.environ["NUMEXPR_MAX_THREADS"] = "16"    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bitwidth", type=int, default=4)
    args = parser.parse_args()
    main(args)
