from rich.console import Console
console = Console()

def viewtensor(filepath: str):
    from triteia.python.utils.tensors import (
        get_tensor_stats,
        tensorstats_to_table,
    )
    stats = get_tensor_stats(filepath)
    table = tensorstats_to_table(filepath, stats)
    console.print(table)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="View tensor stats")
    parser.add_argument("filepath", type=str, help="Path to tensor file")
    args = parser.parse_args()
    viewtensor(args.filepath)
