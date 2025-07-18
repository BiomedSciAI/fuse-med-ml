import hashlib
import inspect
import os
import warnings
from collections.abc import Sequence
from inspect import stack
from typing import Any, Callable, List, Type

from fuse.utils.cpu_profiling import Timer
from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe


def get_function_call_str(
    func: Callable,
    *_args: list,
    _ignore_kwargs_names: List = None,
    _include_code: bool = True,
    **_kwargs: dict,
) -> str:
    """
    Converts a function and its kwargs into a hash value which can be used for caching.

    Note:
    1. This is far from being bulletproof, the op might call another function which is not covered and is changed,
    which will make the caching processing be unaware.
    2. This is a mechanism that helps to spot SOME of such issues, NOT ALL
    3. Only a specific subset of arg types contribute to the caching, mainly simple native python types.
    see 'value_to_string' for more details.
    For example, if an arg is an entire numpy array, it will not contribute to the total hash.
    The reason is that it will make the cache calculation too slow, and might


    _ignore_kwargs_names: args to ignore when generating the string representation. This is useful for kwargs
        that you don't want to have an effect on the hash, for example, verbose flag.
        example usage: ignore_kwargs_names=['verbose', 'cpu_cores_num']

    _include_code: would code be included in the string representation. Note, it will only include code found *directly*
        in the provided function. Set to False if you don't want it to influence the generated string.
        Default is True.


    """
    if _ignore_kwargs_names is None:
        _ignore_kwargs_names = []
    kwargs = convert_func_call_into_kwargs_only(func, *_args, **_kwargs)

    args_flat_str = func.__name__ + "@"
    use_keys = [k for k in sorted(kwargs.keys()) if k not in _ignore_kwargs_names]
    # ignore_kwargs_names
    args_flat_str += "@".join(
        [f"{str(k)}@{value_to_string(kwargs[k])}" for k in use_keys]
    )

    module_str = str(
        inspect.getmodule(func)
    )  # adding full (including scope) name of the function, for the case of multiple functions with the same name
    if " from " in module_str:
        module_str = module_str.split(" from ")[0]
    args_flat_str += "@" + module_str

    if _include_code:
        args_flat_str += "@" + inspect.getsource(
            func
        )  # considering the source code (first level of it...)

    return args_flat_str


def value_to_string(val: Any, warn_on_types: Sequence | None = None) -> str:
    """
    Used by default in several caching related hash builders.
    Ignores <...> string as they usually change between different runs
    (for example, due to pointing to a specific memory address)
    """
    if warn_on_types is not None:
        if isinstance(val, tuple(warn_on_types)):
            warnings.warn(
                f"type {type(val)} is possibly participating in hashing, this is usually not optimal performance wise."
            )
    ans = str(val)
    if ans.startswith("<"):
        return ""
    return str(val)


def convert_func_call_into_kwargs_only(
    func: Callable,
    *args: list,
    **kwargs: dict,
) -> dict:
    """
    Considers positional and kwargs (including their default values !)
    and converts into ONLY kwargs
    """
    signature = inspect.signature(func)

    my_kwargs = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # convert positional args into kwargs
    # uses the fact that zip stops on the smallest length ( so only as much as len(args))
    for curr_pos_arg, pos_arg_name in zip(args, inspect.getfullargspec(func).args):
        my_kwargs[pos_arg_name] = curr_pos_arg

    my_kwargs.update(kwargs)

    return my_kwargs


def get_callers_string_description(
    max_look_up: int,
    expected_class: Type,
    expected_function_name: str,
    value_to_string_func: Callable = value_to_string,
    ignore_first_frames: int = 3,  # one for this function, one for HashableCallable, and one for OpBase
) -> str:
    """
    Iterates on the callstack, and accumulates a string representation of the callers args.
    Used in OpBase to "record" the __init__ args, to be used in the string representation of an Op,
    which is used for building a hash value for samples caching in SamplesCacher

    example call:

    class A:
        def __init__(self):
            text = get_callers_string_description(4, A,

    class B(A):
        def __init__(self, blah, blah2):
            super().__init__()
            #... some logic



    :param max_look_up: how many stack frames to look up
    :param expected_class: what class is the method expected to be,
        stack frames in a different class will be skipped.
        pass None for not requiring any class
    :param expected_function_name: what is the name of the function to allow,
        stack frames in a different function name will be skipped,
        pass None for not requiring anything
    :param value_to_string_func: allows to provide a custom function for converting values to strings
    :param
    """
    str_desc = ""
    try:
        curr_stack = stack()
        curr_locals = None
        # note: frame 0 is this function, frame 1 is whoever called this (and wanted to know about its callers),
        # so both frames 0+1 are skipped.
        for i in range(
            ignore_first_frames, min(len(curr_stack), max_look_up + ignore_first_frames)
        ):
            curr_locals = curr_stack[i].frame.f_locals
            if expected_class is not None:
                if "self" not in curr_locals:
                    continue
                if not isinstance(curr_locals["self"], expected_class):
                    continue

            if expected_function_name is not None:
                if expected_function_name != str(curr_stack[i].function):
                    continue

            curr_str = ".".join(
                [
                    str(
                        curr_locals["self"].__module__
                    ),  # module is probably not needed as class already contains it
                    str(curr_locals["self"].__class__),
                    str(curr_stack[i].function),
                ]
            )

            curr_str += inspect.getsource(curr_stack[i].frame)
            for k, d in curr_stack[i].frame.f_locals.items():
                if "self" == k:
                    continue
                if k.startswith("__"):
                    continue
                curr_str += "@" + str(k) + "@" + value_to_string_func(d)

            str_desc += curr_str

    finally:
        del curr_locals
        del curr_stack

    return str_desc


# TODO: consider adding "ignore list" of args that should not participate in cache value calculation (for example - "verbose")
def run_cached_func(cache_dir: str, func: Callable, *args: list, **kwargs: dict) -> Any:
    """
    Will cache the function output in the first time that
     it is executed, and will load from cache on the next times.

    The cache hash value will be based on the function name, the args, and the code of the function.

    Args:
    :param cache_dir: the directory into which caches will be stored/loaded
    :param func: the function to run
    :param *args: positional args to provide to the function
    :param **kwargs: kwargs to provide to the function

    """
    os.makedirs(cache_dir, exist_ok=True)
    call_str = get_function_call_str(func, *args, **kwargs)
    call_hash = hashlib.md5(call_str.encode("utf-8")).hexdigest()

    cache_full_file_path = os.path.join(cache_dir, call_hash + ".pkl.gz")
    print(f"cache_full_file_path={cache_full_file_path}")

    if os.path.isfile(cache_full_file_path):
        with Timer(f"loading {cache_full_file_path}"):
            ans = load_pickle(cache_full_file_path)
        return ans

    with Timer("run_cached_func::running func ..."):
        ans = func(*args, **kwargs)

    with Timer(f"saving {cache_full_file_path}"):
        save_pickle_safe(ans, cache_full_file_path, compress=True)

    return ans
