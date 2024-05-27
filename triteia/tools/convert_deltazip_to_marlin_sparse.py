import json
import cupy as cp
from tqdm import tqdm
import safetensors as st
import torch, argparse, copy
from triteia.lib.marlin import Layer_2_4 as MarlinLayer
from triteia.utils.io import save_tensors
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.utils.compressor import LosslessCompressor

@torch.no_grad()
def convert_model(args, verbose=True):
    DEV = "cuda:0"
    tensors = {}
    new_tensors = {}
    remaining_keys = []
    with st.safe_open(args.ckpt, framework="torch", device="cuda:0") as f:
        keys = f.keys()
        remaining_keys = list(f.keys())
        metadata = f.metadata()
        for key in keys:
            tensors[key] = f.get_tensor(key)
            if args.lossless:
                tensors_dtypes = json.loads(metadata["dtype"])
                tensors_shapes = json.loads(metadata["shape"])
    if args.lossless:
        with cp.cuda.Device(0):
            for key in tensors.keys():
                tensors[key] = cp.array(tensors[key], copy=False)
        lc = LosslessCompressor()
        tensors = lc.decompress_state_dict(
            tensors,
            tensors_shapes,
            tensors_dtypes,
            use_bfloat16=False,
            target_device="cuda:0",
        )
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
        layer.meta = torch.empty((m, k // 16), dtype=torch.int16, device=DEV)
        layer.s = torch.empty((k_sp // (k // 2), m), dtype=torch.half, device=DEV)
        layer.pack(
            dequantized_weight,
            scales=scales,
            trans=True,
        )
        new_tensors[module + ".qweight"] = layer.B
        new_tensors[module + ".scales"] = layer.s
        new_tensors[module + ".meta"] = layer.meta
        remaining_keys.remove(module + ".qweight")
        remaining_keys.remove(module + ".qzeros")
        remaining_keys.remove(module + ".scales")
        remaining_keys.remove(module + ".g_idx")
    new_tensors.update({key: tensors[key] for key in remaining_keys})
    return new_tensors        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--lossless", action="store_true")
    args = parser.parse_args()
    print("Converting model...")
    new_tensors = convert_model(args, verbose=True)
    save_tensors(new_tensors, args.save_path)