import torch
from timeit import default_timer as timer
from torch.profiler import profile, record_function, ProfilerActivity

def benchmark_functions(functions, args, n_iters=20):
    """Benchmark a list of functions with the same arguments.

    Args:
        functions (list): list of functions to benchmark
        args (list): list of arguments to pass to each function
        n_iters (int): number of iterations to run each function

    Returns:
        list: list of tuples (function name, average time, min time, max time)
    """
    results = []
    for func_id, func in enumerate(functions):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start = timer()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function(str(func)):
                for _ in range(n_iters):
                    func(*args[func_id])
        torch.cuda.synchronize()
        end = timer()
        results.append({
            "name": str(func),
            "avg": 1000 * (end - start) / n_iters,
        })
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return results