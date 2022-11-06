import functools
from typing import Any, List, Optional
from fuse.utils.utils_debug import FuseDebug
import torch
from tqdm import tqdm
import multiprocessing as mp
from termcolor import cprint
import os
import traceback

"""
global dictionary that stores arguments to a new created process
Direct access is not allowed - use get_from_global_storage to access it from a worker function.
call the following "private" functions only if you know what you're doing: _store_in_global_storage, _remove_from_global_storage
Typically, you'd only need to call get_from_global_storage from your worker_func, and call run_multiprocessed and provide copy_global_storage to it,
    with a dict of values that you want accesible from the worker_func.
"""
_multiprocess_global_storage = {}


def run_multiprocessed(
    worker_func,
    args_list,
    workers=0,
    verbose=0,
    copy_to_global_storage: Optional[dict] = None,
    keep_results_order: bool = True,
    as_iterator=False,
    mp_context: Optional[str] = None,
    desc: Optional[str] = None,
) -> List[Any]:
    """
    Args:
        worker_func: a worker function, must accept only a single positional argument and no optional args.
            For example:
            def some_worker(args):
                speed, height, banana = args
                ...
                return ans
        args_list: a list in which each element is the input to func
        workers: number of processes to use. Use 0 for no spawning of processes (helpful when debugging)
        copy_to_global_storage: Optional - to optimize the running time - the provided dict will be stored in a way that is accessible to worker_func.
         calling get_from_global_storage(...) will allow access to it from within any worker_func
        This allows to create a significant speedup in certain cases, and the main idea is that it allows to drastically reduce the amount of data
         that gets (automatically) pickled by python's multiprocessing library.
        Instead of copying it for each worker_func invocation, it will be copied once, upon worker process initialization.
        keep_results_order: determined if imap or imap_unordered is used. if strict_answers_order is set to False, then results will be ordered by their readiness.
         if strict_answers_order is set to True, the answers will be provided at the same order as defined in the args_list
        as_iterator: if True, a lightweight iterator is returned. This is useful in the cases that the entire returned answer doesn't fit in memory.
         or in the case that you want to parallelize some calculation with the generation.
         if False, the answers will be accumulated to a list and returned.
    :param mp_context: "fork", "spawn", "thread" or None for multiprocessing default
    Returns:
        if as_iterator is set to True, returns an iterator.
        Otherwise, returns a list of results from calling func
    """

    iter = _run_multiprocessed_as_iterator_impl(
        worker_func=worker_func,
        args_list=args_list,
        workers=workers,
        verbose=verbose,
        copy_to_global_storage=copy_to_global_storage,
        keep_results_order=keep_results_order,
        mp_context=mp_context,
        desc=desc,
    )

    if as_iterator:
        return iter

    ans = [x for x in iter]
    return ans


def _run_multiprocessed_as_iterator_impl(
    worker_func,
    args_list,
    workers=0,
    verbose=0,
    copy_to_global_storage: Optional[dict] = None,
    keep_results_order: bool = True,
    mp_context: Optional[str] = None,
    desc: Optional[str] = None,
) -> List[Any]:
    """
    an iterator version of run_multiprocessed - useful when the accumulated answer is too large to fit in memory

    Args:
        worker_func: a worker function, must accept only a single positional argument and no optional args.
            For example:
            def some_worker(args):
                speed: height, banana = args
                ...
                return ans
        args_list: a list in which each element is the input to func
        workers: number of processes to use. Use 0 for no spawning of processes (helpful when debugging)
        copy_to_global_storage: Optional - to optimize the running time - the provided dict will be stored in a way that is accessible to worker_func.
            calling get_from_global_storage(...) will allow access to it from within any worker_func
        This allows to create a significant speedup in certain cases, and the main idea is that it allows to drastically reduce the amount of data
            that gets (automatically) pickled by python's multiprocessing library.
        Instead of copying it for each worker_func invocation, it will be copied once, upon worker process initialization.
        keep_results_order: determined if imap or imap_unordered is used. if strict_answers_order is set to False, then results will be ordered by their readiness.
            if strict_answers_order is set to True, the answers will be provided at the same order as defined in the args_list
    :param mp_context: "fork", "spawn", "thread" or None for multiprocessing default
    """
    if "DEBUG_SINGLE_PROCESS" in os.environ and os.environ["DEBUG_SINGLE_PROCESS"] in ["T", "t", "True", "true", 1]:
        workers = None
        cprint(
            "Due to the env variable DEBUG_SINGLE_PROCESS being set, run_multiprocessed is not using multiprocessing",
            "red",
        )

    if FuseDebug().get_setting("multiprocessing") == "main_process":
        workers = None
        cprint("Due to the FuseDebug mode, run_multiprocessed is not using multiprocessing", "red")

    assert callable(worker_func)

    if verbose < 1:
        tqdm_func = lambda x: x
    else:
        tqdm_func = functools.partial(tqdm, desc=desc)

    if copy_to_global_storage is None:
        copy_to_global_storage = {}

    if workers is None or workers <= 1:
        _store_in_global_storage(copy_to_global_storage)
        try:
            for i in tqdm_func(range(len(args_list))):
                curr_ans = worker_func(args_list[i])
                yield curr_ans
        except:
            raise
        finally:
            _remove_from_global_storage(list(copy_to_global_storage.keys()))
    else:
        assert isinstance(workers, int)
        assert workers >= 0

        if mp_context == "thread":
            from multiprocessing.pool import ThreadPool

            pool = ThreadPool
        elif mp_context is None:  # os default
            pool = mp.Pool
        else:
            pool = mp.get_context(mp_context).Pool

        worker_func = functools.partial(worker_func_wrapper, worker_func=worker_func)
        with pool(processes=workers, initializer=_store_in_global_storage, initargs=(copy_to_global_storage,)) as pool:
            if verbose > 0:
                cprint(f"multiprocess pool created with {workers} workers.", "cyan")
            map_func = pool.imap if keep_results_order else pool.imap_unordered
            for curr_ans in tqdm_func(
                map_func(worker_func, args_list), total=len(args_list), smoothing=0.1, disable=verbose < 1
            ):
                yield curr_ans


def worker_func_wrapper(*args, worker_func, **kwargs):
    torch.set_num_threads(1)
    return worker_func(*args, **kwargs)


def _store_in_global_storage(store_me: dict) -> None:
    """
    Copy elements to new pool processes to optimize the running time
    The arguments will be added to a global dictionary multiprocess_copied_args
    :param kwargs: list of tuples - each tuple is a key-value pair and will be added to the global dictionary
    :return: None
    """
    if store_me is None:
        return

    global _multiprocess_global_storage

    # making sure there are no name conflicts
    for key, _ in store_me.items():
        assert (
            key not in _multiprocess_global_storage
        ), f"run_multiprocessed - two multiprocessing pools with num_workers=0 are running simultaneously and using the same argument name {key}"

    _multiprocess_global_storage.update(store_me)


def _remove_from_global_storage(remove_me: List) -> None:
    """
    remove copied args for multiprocess
    :param kwargs: list of tuples - each tuple is a key-value pair and will be added to the global dictionary
    :return: None
    """
    if remove_me is None:
        return

    global _multiprocess_global_storage
    for key in remove_me:
        del _multiprocess_global_storage[key]


def get_from_global_storage(key: str) -> Any:
    """
    Get args copied by run_multiprocessed
    """
    global _multiprocess_global_storage
    return _multiprocess_global_storage[key]


ctx = mp.get_context("spawn")


class Process(ctx.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._start_method = None  # don't force spawn from now on

    def run(self):
        try:
            results = self._target(*self._args, **self._kwargs)
            self._cconn.send((results, None))
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((None, (e, tb)))
            raise e  # You can still rise this exception if you need to

    @property
    def results_and_error(self):
        if self._pconn.poll():
            return self._pconn.recv()
        return (None, None)


def run_in_subprocess(f: callable, *args, timeout: int = 600, **kwargs):
    """A decorator that makes function run in a subprocess.
    This can be useful when you want allocate GPU and memory and to release it when you're done.
    :param f: the function to run in a subprocess
    :param timeout: the maximum time to wait for the process to complete
    """
    if not os.environ.get("FORCE_RUN_IN_SUBPROCESS", True):
        return f(*args, **kwargs)

    # create process
    p = Process(target=f, args=args, kwargs=kwargs)  # using a subclass of Process
    p.start()
    try:
        p.join(timeout=timeout)
    except:
        p.terminate()
        raise

    results, error = p.results_and_error
    if error is not None:
        exception, traceback = error
        print(f"process func {f} had an exception: {exception}")
        print(traceback)
        raise RuntimeError(f"process func {f} had an exception: {exception}")

    assert p.exitcode == 0, f"process func {f} failed with exit code {p.exitcode}"
    return results
