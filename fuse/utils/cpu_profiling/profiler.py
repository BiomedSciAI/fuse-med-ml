import cProfile
import pstats
from io import StringIO


class Profiler:
    """
    A simple to use cpu profiler, helps to find CPU bottlenecks

    all you need to do is wrap the code you want to analyze:

    with Profiler('blahblah'):
        #... some code

    Usage example:

    #we create two functions, one is quick and one is very slow

    from itertools import combinations
    def dummy_slow():
        N = 140 #1000
        K = 2
        for _ in range(100):
            all_combs = combinations(range(N),K)
            all_combs = list(all_combs)
        return all_combs

    def dummy_quick():
        a = 1000
        a+=1
        return a

    def get_combs():
        for _ in range(10):
            x = dummy_slow()
            y = dummy_quick()

        return x,y

    with Profiler('banana'):
        get_combs()

    #results in

    starting profiling for banana...
    profiling results for banana :
            23 function calls in 0.779 seconds

    Ordered by: cumulative time

    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1    0.001    0.001    0.779    0.779 /some/path/test_profiler.py:22(get_combs)
    10   0.778    0.078    0.778    0.078 /some/path/test_profiler.py:9(dummy_slow)
    1    0.000    0.000    0.000    0.000 /some/path/profiler.py:45(__exit__)
    10   0.000    0.000    0.000    0.000 /some/path/test_profiler.py:17(dummy_quick)
    1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    #analyzing it we can see that dummy_slow is the bottelneck within get_combs,
    as its cumulative time is 0.778 secs out of the total of 0.779 seconds of get_combs
    on the other hand, dummy_quick is clearly not the bottleneck.

    """

    def __init__(self, txt: str = ""):
        self.txt = txt

    def __enter__(self):
        print(f"starting profiling for {self.txt}...")
        self.pr = cProfile.Profile()
        self.pr.enable()

    def __exit__(self, *args):
        self.pr.disable()
        print(f"profiling results for {self.txt} :")
        self.s = StringIO()
        sortby = "cumulative"
        self.ps = pstats.Stats(self.pr, stream=self.s).sort_stats(sortby).reverse_order()
        self.ps.print_stats()
        print(self.s.getvalue())
