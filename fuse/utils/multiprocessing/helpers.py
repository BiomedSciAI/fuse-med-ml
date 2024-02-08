import numpy as np
from typing import List, Tuple
import os


def num_available_cores(verbose: bool = True) -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            ans = len(os.sched_getaffinity(0))
            if verbose:
                print(
                    f"num_available_cores:: spotted affinity which restricts available cores. Returning {ans} cores"
                )
            return ans
        except Exception:
            pass

    ans = os.cpu_count()
    if verbose:
        print(f"num_available_cores:: Returning {ans} cores")
    return ans


def get_chunks_ranges(
    total: int, *, chunk_size: int = None, parts: int = None
) -> List[Tuple[int, int]]:
    """
    Creates "chunks" of work, useful when creating worker functions for run_multiprocessed
     where each workers gets a range ("chunk") to work on.

    Returns a list of tuples [start_index, end_index)

    Args:
        chunk_size - what is the size of a single chunks
        parts: how many chunks are desired

    for example: get_chunks_ranges(10, chunk_size = 3,) would return [(0,3), (3,6), (6,9), (9,10)]
        and get_chunks_ranges(10, parts=2) would return [(0,5),(5,10)]
    """

    assert (chunk_size is not None) ^ (
        parts is not None
    ), "Exactly one of chunk_size or parts must be provided"

    if chunk_size is not None:
        if chunk_size >= total:
            return [(0, total)]

        steps = np.arange(0, total, chunk_size, dtype=np.int64)
        ans = list(zip(steps[:-1], steps[1:]))
        if ans[-1][-1] < total:
            ans.append((ans[-1][-1], total))
        ans[-1] = (ans[-1][0], min(ans[-1][-1], total))
        return ans

    elif parts is not None:
        chunk_size = np.ceil(total / parts)
        return get_chunks_ranges(total, chunk_size=chunk_size)

    assert False, "should not reach here"


if __name__ == "__main__":
    ans = get_chunks_ranges(12, parts=2)
    z = 1
