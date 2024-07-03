from pynvml import *
nvmlInit()

nvidia_rtx_3090 = {
    'name': 'NVIDIA GeForce RTX 3090',
    'compute_capability': '8.6',
    'memory': 24, # in GB
    'bandwidth': 936.2,
    'fp16_tflops': 35.58,
    'fp32_tflops': 35.58,
}

nvidia_gpus = [nvidia_rtx_3090]

def get_gpu_device_info():
    deviceCount = nvmlDeviceGetCount()
    name = None
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        break
    for gpu in nvidia_gpus:
        if gpu['name'] == name:
            return gpu
    return None