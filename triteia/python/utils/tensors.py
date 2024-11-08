import json
import safetensors as st
from rich.table import Table
from safetensors.torch import save_file

def get_tensor_stats(filepath):
    tensor_stats = {}
    with st.safe_open(filepath, "torch") as f:
        for key in f.keys():
            tensor_stats[key] = {
                "shape": f.get_tensor(key).shape,
                "dtype": f.get_tensor(key).dtype,
                "nbytes": f.get_tensor(key).nbytes,
            }
    return tensor_stats


def tensorstats_to_table(filepath, tensor_stats):
    table = Table(title=f"Tensor Stats <{filepath}>")
    table.add_column("Key")
    table.add_column("Shape")
    table.add_column("Dtype")
    table.add_column("Kilobytes")
    table.add_column("Range")
    for key in tensor_stats:
        table.add_row(
            key,
            str(tensor_stats[key]["shape"]),
            str(tensor_stats[key]["dtype"]),
            str(tensor_stats[key]["nbytes"] / 1024),
        )
    return table
