import torch
from triteia.python.utils.io import read_tensors
from triteia.python.ops import matmul_4bit_2_4
from triteia.python.ops.utils.generator import generate_model_distribution

def check_tp_group_equal(weights, reference_weights):
    modules = set()
    tp_groups = set()
    
    for key in weights.keys():
        # separate by .
        # last element - component, second last - tp id, others - module name
        tp_group = key.split(".")[-2]
        tp_groups.add(tp_group)
        module_name = ".".join(key.split(".")[:-2])
        modules.add(module_name)
    
    for module in modules:
        tp_groups_in_modules = max([int(key.split(".")[-2]) for key in weights.keys() if module in key]) + 1
        components_in_modules = [key.split(".")[-1] for key in weights.keys() if module in key]
        for component in components_in_modules:
            components_across_tp = [value for key, value in weights.items() if module in key and component in key]
            # there should be at most tp_groups_in_modules tensors for each component
            assert len(components_across_tp) == tp_groups_in_modules, f"Module {module} has {len(components_across_tp)} components for {component}"
            # check if there are same tensors for each component
            for i in range(1, len(components_across_tp)):
                if torch.equal(components_across_tp[i-1], components_across_tp[i]):
                    print(f"Module {module} has same tensors for {component} in tp group {i-1} and {i}")
            
def check_output(weights, reference_weights, module_name):
    target_weight = {key: value for key, value in weights.items() if module_name in key}
    reference_weight = {key: value for key, value in reference_weights.items() if module_name in key}
    tp_groups = set()
    for key in weights.keys():
        # separate by .
        # last element - component, second last - tp id, others - module name
        tp_group = key.split(".")[-2]
        tp_groups.add(tp_group)
    reference_qweight = reference_weights[f"{module_name}.0.qweight"]
    reference_meta = reference_weights[f"{module_name}.0.meta"]
    reference_scale = reference_weights[f"{module_name}.0.scales"]
    
    nr = 10
    x = torch.randn((nr, 32 * reference_qweight.size(0)), dtype=torch.float16, device='cuda')
    reference_output = matmul_4bit_2_4(reference_qweight, x, reference_meta, reference_scale)
    tp_outputs = []
    tp_groups = sorted(list(tp_groups))
    for tp in tp_groups:
        qweight = target_weight[f"{module_name}.{tp}.qweight"]
        meta = target_weight[f"{module_name}.{tp}.meta"]
        scale = target_weight[f"{module_name}.{tp}.scales"]
        output = matmul_4bit_2_4(qweight, x, meta, scale)
        tp_outputs.append(output)
    tp_output = torch.cat(tp_outputs, dim=1)
    
    print(f"reference_output: {reference_output.shape}, tp_output: {tp_output.shape}")
    print(f"first half reference_out: \n{reference_output[:, :reference_output.size(1)//2]}\nfirst half tp_out: \n{tp_output[:, :tp_output.size(1)//2]}")
    
    print(f"second half reference_out: \n{reference_output[:, reference_output.size(1)//2:]}\nsecond half tp_out: \n{tp_output[:, tp_output.size(1)//2:]}")

    print(f"reference_output: \n{reference_output}\ntp_output: \n{tp_output}")
    
    print(f"max diff: {torch.max(torch.abs(reference_output - tp_output))}")
    
def verify(args):
    print(args)
    weights = read_tensors(args.input, device='cuda')
    reference_weights = read_tensors(args.reference_input, device='cuda')
    check_output(weights, reference_weights, "model.layers.9.self_attn.qkv_proj")
    # check_tp_group_equal(weights, reference_weights)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to the input file")
    parser.add_argument("--reference-input", default="", type=str, help="Path to the input file")
    args = parser.parse_args()
    verify(args)