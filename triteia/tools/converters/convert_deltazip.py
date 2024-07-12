import json
import cupy as cp
from tqdm import tqdm
import safetensors as st
import torch, argparse
from triteia.python.utils.io import save_tensors
from triteia.python.utils.quant_utils import dequantize_weight
from triteia.python.utils.compressor import LosslessCompressor
from triteia.python.configs.models.llama import row_chunking_modules, uncompressed_row_chunking_modules, pack_modules
from triteia.python.nn.linear import sparse_low_precision_linear

@torch.no_grad()
def torch_weight_to_sparse_marlin(weight, scale, tp_size=1, chunk_by="column"):
    """
    Args:
        weight: torch.Tensor of shape (in_features, out_features)
        scale: torch.Tensor of shape (1, out_features)
        tp_size: tensor parallelism size
        chunk_by: "column" or "row"
    """
    assert chunk_by in ["column", "row"], "chunk_by must be either 'column' or 'row'"
    assert weight.dim() == 2, "weight must be a 2D tensor"
    assert weight.size(0) % tp_size == 0, "out_features must be divisible by tp_size"
    assert weight.size(1) == scale.size(1), "out_features of weight and scale must match"
    
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    
    qweights, scales,metas = [], [], []
    for i in range(tp_size):
        if chunk_by == "column":
            tp_weight = weight[
                :, 
                i * weight.size(1) // tp_size: (i + 1) * weight.size(1) // tp_size
            ]
            tp_scales = scale[
                :, 
                i * scale.size(1) // tp_size: (i + 1) * scale.size(1) // tp_size
            ]
        elif chunk_by == "row":
            tp_weight = weight[
                i * weight.size(0) // tp_size: (i + 1) * weight.size(0) // tp_size, 
                :
            ]
            tp_scales = scale
        layer = sparse_low_precision_linear(
            infeatures=tp_weight.size(0),
            outfeatures=tp_weight.size(1),
            groupsize=-1
        )
        k, m = tp_weight.size(0), tp_weight.size(1)
        k_sp = k // 2
        layer.groupsize = k
        layer.B = torch.empty((k_sp // 16, m * 16 // 8), dtype=torch.int)
        layer.meta = torch.empty((m, k // 16), dtype=torch.int16)
        layer.s = torch.empty((k_sp // (k // 2), m), dtype=torch.half)
        layer.pack(tp_weight, scales=tp_scales, trans=True)
        qweights.append(layer.B)
        scales.append(layer.s)
        metas.append(layer.meta)
    return qweights, scales, metas

@torch.no_grad()
def convert_model(args, verbose=True):
    DEV = "cuda:0"
    
    new_tensors = {}
    tensors = {}
    packed_tensors = {}
    dequantized_tensors = {}
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
        print(f"Decompressing from lossless format...")
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
    # infeatures, outfeatures
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    pbar = tqdm(quantized_modules, position=0, leave=True)
    print("Dequantizing weights...")
    for module in pbar:
        dequantized_weight = dequantize_weight(
            tensors[module + ".qweight"],
            tensors[module + ".qzeros"],
            tensors[module + ".scales"],
        ).to(torch.float16).t().cpu()
        scales = tensors[module + ".scales"]
        dequantized_tensors[module] = (dequantized_weight, scales)
        remaining_keys.remove(module + ".qweight")
        remaining_keys.remove(module + ".qzeros")
        remaining_keys.remove(module + ".scales")
        remaining_keys.remove(module + ".g_idx")
        
    # now start to pack weights together
    pack_plan = {}
    for module in quantized_modules:
        if any([key in module for key in pack_modules.keys()]):
            source_layer = module.rsplit(".", 2)[0]
            source_module = module.replace(source_layer+".", "")
            target_module = pack_modules[source_module]
            target_idx = int(target_module.split(":")[1])
            target_module = source_layer + "." + target_module.split(":")[0]
            if target_module not in pack_plan:
                pack_plan[target_module] = []
            pack_plan[target_module].append((module, target_idx))
        
        elif any([key in module for key in row_chunking_modules]):
            qweights, scales, metas = torch_weight_to_sparse_marlin(
                dequantized_tensors[module][0].to(DEV),
                dequantized_tensors[module][1].to(DEV),
                tp_size=args.tp_size,
                chunk_by="row",
            )
            for idx, (qweight, scales, meta) in enumerate(zip(qweights, scales, metas)):
                new_tensors[module + f".{idx}.qweight"] = qweight
                new_tensors[module + f".{idx}.scales"] = scales
                new_tensors[module + f".{idx}.meta"] = meta
    for key in pack_plan.keys():
        key_weights = []
        key_scales = []
        plan = sorted(pack_plan[key], key=lambda x: x[1])
        print(f"Plan for {key}: {plan}")
        for module, idx in plan:
            weight, scales = dequantized_tensors[module]
            assert weight.shape[1] == scales.shape[1]
            key_weights.append(weight)
            key_scales.append(scales)
        key_weights = torch.cat(key_weights, dim=1)
        key_scales = torch.cat(key_scales, dim=1)
        packed_tensors[key] = (key_weights, key_scales)
        torch.cuda.synchronize()
        del dequantized_tensors[module]
        torch.cuda.empty_cache()
        
        qweights, scales, metas = torch_weight_to_sparse_marlin(
            packed_tensors[key][0].to(DEV),
            packed_tensors[key][1].to(DEV),
            tp_size=args.tp_size,
            chunk_by="column",
        )
        for idx, (qweight, scales, meta) in enumerate(zip(qweights, scales, metas)):
            new_tensors[key + f".{idx}.qweight"] = qweight
            new_tensors[key + f".{idx}.scales"] = scales
            new_tensors[key + f".{idx}.meta"] = meta
    
    # # now processing remaining keys
    for module in remaining_keys:
        if any([key in module for key in uncompressed_row_chunking_modules]):
            weight = tensors[module]
            module_name = module.removesuffix(".weight")
            num_rows = weight.shape[0]
            for i in range(args.tp_size):
                tp_weight = weight[i * num_rows // args.tp_size: (i + 1) * num_rows // args.tp_size, :]
                new_tensors[module_name + f".{i}.weight"] = tp_weight
    
    return new_tensors
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()
    
    print("Converting model...")
    new_tensors = convert_model(args, verbose=True)
    save_tensors(new_tensors, args.save_path)