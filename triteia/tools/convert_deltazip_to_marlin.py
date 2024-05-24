import gc
import safetensors as st
import torch, argparse, copy
from marlin import Layer as MarlinLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
from triteia.ao.utils.quant_utils import dequantize_weight
from tqdm import tqdm
@torch.no_grad()
def convert_model(args):
    tensors = {}
    with st.safe_open(args.ckpt, framework="pt") as f:
        remaining_keys = list(f.keys())
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    for module in tqdm(quantized_modules):
        
    
    
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        if verbose:
            print(f"--- Converting Module: {name}")
        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1:]

        # Dequantize the weight.
        dequantized_weight = dequantize_weight(module).to(torch.float16)
        linear_module = torch.nn.Linear(
            in_features=dequantized_weight.shape[1],
            out_features=dequantized_weight.shape[0],
            bias=False,
            dtype=torch.float16,
            device="cuda")
        linear_module.weight.data.copy_(dequantized_weight)

        # Create new linear method and copy to model.
        new_module = MarlinLayer(
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            groupsize=model.config.quantization_config.group_size)
        new_module.pack(linear_module, scales=copy.deepcopy(module.scales.data.t()))

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del dequantized_weight, module
        torch.cuda.empty_cache()
        gc.collect()

    return model

@torch.no_grad()
def dequantize_model(model, verbose=True):
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        if verbose:
            print(f"--- Dequantizing Module: {name}")
        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1:]

        # Dequantize the weight.
        dequantized_weight = dequantize_weight(module)
        dequantized_weight_cpu = dequantized_weight.to("cpu")

        # Create new linear method and copy to model.
        new_module = torch.nn.Linear(
            in_features=dequantized_weight_cpu.shape[1],
            out_features=dequantized_weight_cpu.shape[0],
            bias=False,
            dtype=torch.float16)
        new_module.weight.data.copy_(dequantized_weight_cpu)
        new_module.scales = torch.nn.Parameter(copy.deepcopy(module.scales.data))

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del dequantized_weight, dequantized_weight_cpu, module
        torch.cuda.empty_cache()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--do-generation", action="store_true")

    args = parser.parse_args()
    model_id = args.model_id
    save_path = args.save_path
    do_generation = args.do_generation

    print("Loading gptq model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Dequantize the Model.
    print("Converting model...")
    model = convert_model(model).to("cpu")

    # Save after updating quantization config.
    print("Saving marlin model...")
    model.config.quantization_config = {
        "group_size": model.config.quantization_config.group_size,
        "quant_method": "marlin"
    }
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if do_generation:
        print("Generating sample text...")
        model.to("cuda")
        prompt = "My favorite song is"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        print(tokenizer.batch_decode(outputs)[0])