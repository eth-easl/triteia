import safetensors as st
from safetensors.torch import save_file

def save_tensors(tensors, path):
    for key in tensors.keys():
        tensors[key] = tensors[key].contiguous()
    save_file(tensors, path)

def read_tensors(path, prefix=None, device='cpu'):
    tensors = {}
    with st.safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            if prefix is None:
                tensors[key] = f.get_tensor(key)
            else:
                if key.startswith(prefix):
                    module_name = key.removeprefix(prefix + ".")
                    tensors[module_name] = f.get_tensor(key)
    return tensors