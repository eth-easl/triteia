from functools import lru_cache


@lru_cache(1)
def warn_once(msg: str):
    print(f"[TK] {msg}")


def vprint(msg: str, verbose=True):
    if verbose:
        print(f"[TK] {msg}")
