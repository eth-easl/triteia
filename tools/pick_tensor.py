import os
import json
import cupy as cp
import safetensors as st
from safetensors.torch import save_file

def main(args):
    print(args)
    tensors = {}
    with st.safe_open(args.ckpt, framework="torch", device="cuda:0") as f:
        metadata = f.metadata()
        keys = f.keys()
        for key in keys:
            tensors[key] = f.get_tensor(key)
    tensors = {
        key: tensors[key] for key in tensors.keys() if args.keyword in key
    }
    # save the decompressed tensors
    save_file(
        tensors,
        args.output,
    )
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to the compressed file")
    parser.add_argument("--keyword", type=str, help="Keyword to search for in the compressed file")
    parser.add_argument("--output", type=str, help="Path to the output file")
    args = parser.parse_args()
    main(args)
    