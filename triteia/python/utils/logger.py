from functools import lru_cache


@lru_cache(1)
def warn_once(msg: str):
    print(f"\cm\033[1;31m{msg}\033[0m")
