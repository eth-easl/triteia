from pynvml import *

nvmlInit()

# https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
nvidia_rtx_3090 = {
    "name": "NVIDIA GeForce RTX 3090",
    "compute_capability": "8.6",
    "memory": 24,  # in GB
    "bandwidth": 936.2,  # in GB/s
    "fp16_tflops": 71,
    "fp32_tflops": 35.58,
}

nvidia_gh200_120gb = {
    "name": "GH200 120GB",
    "compute_capability": "9.0",
    "memory": 120,  # in GB
    "bandwidth": 3350,  # in GB/s
    "fp16_tflops": 989.4,
    "fp32_tflops": 494.7,
}

nvidia_gpus = [nvidia_rtx_3090, nvidia_gh200_120gb]


def get_gpu_device_info():
    deviceCount = nvmlDeviceGetCount()
    name = None
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
    for gpu in nvidia_gpus:
        if gpu["name"] == name:
            return gpu
    return None
