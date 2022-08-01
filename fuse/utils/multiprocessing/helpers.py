import numpy as np
from typing import List, Tuple


def get_chunks_ranges(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    """
    Creates "chunks" of work, useful when creating worker functions for run_multiprocessed
     where each workers gets a range ("chunk") to work on.

    Returns a list of tuples [start_index, end_index)

    for example: get_chunks_ranges(10, 3) would return [(0,3), (3,6), (6,9), (9,10)]
    """

    if chunk_size >= total:
        return [(0, total)]

    steps = np.arange(0, total, chunk_size)
    ans = list(zip(steps[:-1], steps[1:]))
    if ans[-1][-1] < total:
        ans.append((ans[-1][-1], total))
    return ans
