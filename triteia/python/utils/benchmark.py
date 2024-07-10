import json
import torch
import inspect
import pandas as pd
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
    table.caption = f"Tested on {get_gpu_device_info()['name']}"
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

def export_benchmark_results(results, filepath:str):
    gpu_specs = get_gpu_device_info()
    config_results = []
    for result in results:
        for res in result:
            # ignore args if it is torch tensor
            config = {k: v for k, v in res['args'].items() if not isinstance(v, torch.Tensor)}
            del res['args']
            del res['output']
            config_results.append({
                'config': config,
                **res
            })
    with open(filepath, 'w') as f:
        json.dump({
            'gpu_specs': gpu_specs,
            'results': config_results
        }, f, indent=4)

def format_benchmark_results(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    gpu_specs = data['gpu_specs']
    results = data['results']
    df_results = []
    parsed_results = []
    configs = []
    for result in results:
        config = result['config'].copy()
        del result['config']
        res = result.copy()
        func_name = res['func_name']
        del res['func_name']
        res = {f"{func_name}_{k}": v for k, v in res.items()}
        parsed_results.append({
            "config": config,
            **res
        })
        configs.append(config)
    for config in configs:
        res = [d for d in parsed_results if d['config'] == config]
        res = [{k: v for k, v in d.items() if k != 'config'} for d in res]
        results = {}
        for r in res:
            results.update(r)
        df_results.append({
            **config,
            **results,
        })
    df = pd.DataFrame(df_results)
    # deduplicate rows
    df = df.drop_duplicates()
    return gpu_specs, df