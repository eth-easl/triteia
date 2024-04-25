import safetensors as st

def main(args):
    tensors = {}
    with st.safe_open(args.ckpt, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    for key, tensor in tensors.items():
        print(f"{key}: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")
    
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    main(args)
    
    