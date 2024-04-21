import safetensors as st
from ao.utils.bitblas import convert_to_bitblas
from safetensors.torch import save_file


def main(args):
    print(args)
    tensors = {}
    new_tensors = {}
    with st.safe_open(args.ckpt, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    quantized_modules = [
        x.removesuffix(".qweight") for x in tensors.keys() if "qweight" in x
    ]
    for module in quantized_modules:
        qweight, scales, zeros, bias = convert_to_bitblas(
            args.bitwidth, module, tensors
        )
        new_tensors[module + ".qweight"] = qweight
        new_tensors[module + ".scales"] = scales
        new_tensors[module + ".zeros"] = zeros
        if bias is not None:
            new_tensors[module + ".bias"] = bias
    save_file(new_tensors, args.output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--bitwidth", type=int, required=True)

    args = parser.parse_args()
    main(args)
