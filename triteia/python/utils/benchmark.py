import inspect
import torch
from rich.console import Console
from rich.table import Table
from triteia.python.configs.gpus.specs import get_gpu_device_info


def timing_function(func, flops_func, kwargs, repeats=1):
    func_args_names = inspect.getfullargspec(func).args
    func_args = {arg: kwargs[arg] for arg in func_args_names if arg in kwargs}
    gpu_info = get_gpu_device_info()
    if flops_func:
        flops_func_args_names = inspect.getfullargspec(flops_func).args
        flops_func_args = {
            arg: kwargs[arg] for arg in flops_func_args_names if arg in kwargs
        }
    elapseds = []

    for i in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = func(**func_args)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        elapseds.append(elapsed)

    elapsed = sum(elapseds) / repeats

    if flops_func:
        total_flops = flops_func(**flops_func_args)  # FLOPS
        perf_flops = total_flops / elapsed  # FlOPS/ms
        # total_tflops = total_flops/1e12 # TFLOPS
        if gpu_info:
            mfu = 100 * perf_flops / 1e9 / gpu_info["fp16_tflops"]

    return {
        "output": output,
        "elapsed": elapsed,  # ms
        "func_name": func.__name__,
        "total_flops": total_flops / 1e9 if flops_func else None,  # GFLOPS
        "perf_flops": perf_flops / 1e6 if flops_func else None,  # GFLOPS/s
        "mfu": mfu if flops_func and gpu_info else None,
        "args": kwargs,
    }


def print_results_table(title, results):
    table = Table(title=title)
    table.add_column("Func Name")
    table.add_column("Elapsed (ms)")
    table.add_column("Total FLOPS (GFLOPS)")
    table.add_column("Perf FLOPS (GFLOPS/s)")
    table.add_column("MFU (%)")
    for result in results:
        table.add_row(
            result["func_name"],
            f"{result['elapsed']:.2f}",
            f"{result['total_flops']:.2f}" if result["total_flops"] else None,
            f"{result['perf_flops']:.2f}" if result["perf_flops"] else None,
            f"{result['mfu']:.2f}" if result["mfu"] else None,
        )
    console = Console()
    console.print(table)
