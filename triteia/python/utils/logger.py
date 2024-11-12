from functools import lru_cache


@lru_cache(1)
def warn_once(msg: str):
    print(f"[Triteia Kernel] {msg}")
