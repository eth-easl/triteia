import json
import cupy as cp
from tqdm import tqdm
import safetensors as st
import torch, argparse
from triteia.utils.io import save_tensors
from triteia.ao.utils.quant_utils import dequantize_weight
from triteia.utils.compressor import LosslessCompressor
from triteia.utils.conversions import torch_weight_to_sparse_marlin
# NOTE: This is only for llama-series models
column_chunking_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
]
pack_modules = {
    "self_attn.q_proj": "self_attn.qkv_proj:0",
    "self_attn.k_proj": "self_attn.qkv_proj:1",
    "self_attn.v_proj": "self_attn.qkv_proj:2",
    "mlp.gate_proj": "mlp.gate_up_proj:0",
    "mlp.up_proj": "mlp.gate_up_proj:1",
}
row_chunking_modules = [
    "self_attn.o_proj",
    "mlp.down_proj",
]
uncompressed_row_chunking_modules = [
    "embed_tokens",
    "lm_head",
]

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
    for module in pbar:
        dequantized_weight = dequantize_weight(
            tensors[module + ".qweight"],
            tensors[module + ".qzeros"],
            tensors[module + ".scales"],
        ).to(torch.float16).t()
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
    
    for key in pack_plan.keys():
        pack_plan[key] = sorted(pack_plan[key], key=lambda x: x[1])
        first_module_weight = dequantized_tensors[pack_plan[key][0][0]][0]
        first_module_scales = dequantized_tensors[pack_plan[key][0][0]][1]
        packed_tensors[key] = (
            torch.zeros(
                first_module_weight.shape[0],
                sum([dequantized_tensors[module[0]][0].shape[1] for module in pack_plan[key]]),
                dtype=torch.float16,
                device=DEV
            ),
            torch.zeros(
                first_module_scales.shape[0],
                sum([dequantized_tensors[module[0]][1].shape[1] for module in pack_plan[key]]),
                dtype=torch.float16,
                device=DEV
            )
        )
    
    for key in pack_plan.keys():
        print(f"{key}, shape: {packed_tensors[key][0].shape}, {packed_tensors[key][1].shape}")
        start = 0
        for module, idx in pack_plan[key]:
            weight, scales = dequantized_tensors[module]
            print(f"packing {module}, shape: {weight.shape}, {scales.shape}")
            assert weight.shape[1] == scales.shape[1]
            packed_tensors[key][0][
                :,
                start:start + weight.shape[1],
            ].copy_(weight)
            packed_tensors[key][1][
                :,
                start:start + scales.shape[1],
            ].copy_(scales)
            start += weight.shape[1]
        qweights, scales, metas = torch_weight_to_sparse_marlin(
            packed_tensors[key][0],
            packed_tensors[key][1],
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