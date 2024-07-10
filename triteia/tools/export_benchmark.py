import os
from triteia.python.utils.benchmark import format_benchmark_results

def export(args):
    gpu_spec, df = format_benchmark_results(args.in_path)
    # get filename from path
    filename = os.path.basename(args.in_path)
    out_path = os.path.join(args.out_path, f"{filename}_{gpu_spec['name']}.csv")
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Export benchmark results')
    parser.add_argument('--in-path', type=str, help='Filepath to the benchmark results')
    parser.add_argument('--out-path', type=str, help='Filepath to the exported benchmark results')
    args = parser.parse_args()
    export(args)