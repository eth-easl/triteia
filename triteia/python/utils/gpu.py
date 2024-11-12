import torch


def get_gpu_prop():
    properties = torch.cuda.get_device_properties(torch.cuda.device(0))
    return properties


def is_hopper():
    properties = get_gpu_prop()
    if properties.major == 9:
        return True
    return False


def is_ampere():
    properties = get_gpu_prop()
    if properties.major == 8:
        return True
    return False


if __name__ == "__main__":
    prop = get_gpu_prop()
    print(f"GPU prop: {prop}")
    print(f"Is Hopper: {is_hopper()}")
