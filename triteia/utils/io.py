from safetensors.torch import save_file

def save_tensors(tensors, path):
    for key in tensors.keys():
        tensors[key] = tensors[key].contiguous()
    save_file(tensors, path)