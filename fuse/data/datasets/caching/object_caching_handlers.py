from typing import List, Any
import numpy as np
from fuse.utils.ndict import NDict
import torch

# TODO: support custom _object_requires_hdf5_single
#      maybe even more flexible (knowing key name etc., patterns, explicit name, regular expr.)

# TODO: should we require OrderedDict?? and for the internal dicts as well ??
# TODO: maybe it's better to flatten the dictionaries first


def _object_requires_hdf5_recurse(curr: NDict, str_base: str = "") -> List[str]:
    """
    Iterates on keys and checks
    """
    keys = curr.keypaths()
    ans = []
    for k in keys:
        data = curr[k]
        if _object_requires_hdf5_single(data):
            ans.append(k)
    return ans


# def _PREV__object_requires_hdf5_recurse(curr: NDict, str_base='') -> List[str]:
#     """
#     Recurses (only into dicts!) and returns a list of keys that require storing into HDF5
#     (which allows reading only sub-parts)

#     :return: a list of keys as strings, e.g. ['data.cc.img', 'data.mlo.img']
#     """
#     #print('str_base=', str_base)
#     if _object_requires_hdf5_single(curr):
#         return str_base

#     if isinstance(curr, dict):
#         ans = []
#         for k,d in curr.items():
#             curr_ans = _object_requires_hdf5_recurse(
#                 d, str_base+'.'+k if str_base!='' else k,
#             )
#             if curr_ans is None:
#                 pass
#             elif isinstance(curr_ans, list):
#                 ans.extend(curr_ans)
#             else:
#                 ans.append(curr_ans)
#         return ans

#     return None


_error_msg_torch_tensors = "You need to cast to numpy ndarray in the dynamic pipeline as it takes a lot of time pickling torch.Tensor"


def _object_requires_hdf5_single(obj, minimal_ndarray_size=100) -> bool:  # type: ignore
    if _valid_ndarray(obj, minimal_ndarray_size):
        return True

    if isinstance(obj, torch.Tensor):
        raise Exception(_error_msg_torch_tensors)

    if isinstance(obj, (list, tuple)):
        if any(
            [_valid_ndarray(x, minimal_ndarray_size) for x in obj]
        ):  # _valid_ndarray(obj[0], minimal_ndarray_size):
            assert all(
                [isinstance(x, np.ndarray) for x in obj]
            ), f"first element in {type(obj)} is a numpy array but not all of the rest are! This is not supported."
            return True
        if any([torch.is_tensor(x) for x in obj]):
            raise Exception(_error_msg_torch_tensors)

    # if ans:
    #    print(f'found hfd5 requiring object! shape={obj.shape}, size={obj.size}')
    return False


def _valid_ndarray(obj: Any, minimal_ndarray_size: int) -> bool:
    return isinstance(obj, np.ndarray) and (obj.size > minimal_ndarray_size)
