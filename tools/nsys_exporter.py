import os
from tqdm import tqdm
local_path = ".local/run_4"
kernels = ['ibmm_fp16_for', 'ibmm_fp16_bmm', 'for-loop', 'ibmm_native']
Ks = [2048, 4096]
Ms = [2048, 4096]
num_requests = [100]
num_models = [16,64]

trials = 5
jobs = []
for i in range(trials):
    for kernel in kernels:
        for M, K in zip(Ms, Ks):
            size = '2k' if M == 2048 else '4k'
            for num_req in num_requests:
                for num_model in num_models:
                    job = f"nsys stats -r cuda_kern_exec_sum --filter-nvtx '{i} {kernel} {num_model}x{M}x{K}' {local_path}/run_4.nsys-rep -f json --output {local_path}/{size}_{kernel}_{i}_{num_model}.json"
                    jobs.append(job)
for job in tqdm(jobs):
    os.system(job)