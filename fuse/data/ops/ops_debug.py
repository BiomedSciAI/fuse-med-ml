from typing import Any, Hashable, List, Sequence, Optional
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.data import OpBase
import numpy
import torch

class OpDebugBase(OpBase):
    """Base class for debug operations"""
    def __init__(self, sample_ids: Optional[List[Hashable]] = None, first_sample_only: bool = False):
        """
        :param sample_ids: apply for the specified sample ids. To apply for all set to None.
        :param first_sample_only: apply for the first sample only
        """
        super().__init__()
        self._sample_ids = sample_ids
        self._first_sample_only = first_sample_only
        self._first_sample_done = False

    def should_debug_sample(self, sample_dict: NDict) -> bool:
        if self._first_sample_only and self._first_sample_done:
            return False
        
        if self._sample_ids is not None:
            sid = get_sample_id(sample_dict)
            if sid not in self._sample_ids:
                return False
        
        self._first_sample_done = True
        return True

class OpPrintKeys(OpDebugBase):
    """
    Print list of available keys at a given point in the data pipeline
    It's recommended, but not a must, to run it in a single process.
    Add at the top your script to force single process:
    ```
    from fuse.utils.utils_debug import FuseDebug
    FuseDebug("debug")
    ```

    Example:
    ```
    (OpPrintKeys(first_sample_only), dict()),
    ```
    """
    def __call__(self, sample_dict: NDict) -> Any:
        if self.should_debug_sample(sample_dict):
            print(f"Sample {get_sample_id(sample_dict)} keys:")
            for key in sample_dict.keypaths():
                print(f"{key}")
        return sample_dict

class OpPrintShapes(OpDebugBase):
    """
    Print the shapes/length of every torch tensor / numpy array / sequence 
    Add at the top your script to force single process:
    ```
    from fuse.utils.utils_debug import FuseDebug
    FuseDebug("debug")
    ```
    Example:
    ```
    (OpPrintShapes(first_sample_only), dict()),
    ```
    """
    def __call__(self, sample_dict: NDict) -> Any:
        if self.should_debug_sample(sample_dict):
            print(f"Sample {get_sample_id(sample_dict)} shapes:")
            for key in sample_dict.keypaths():
                value = sample_dict[key]
                if isinstance(value, torch.Tensor):
                    print(f"{key} is tensor with shape: {value.shape}")
                elif isinstance(value, numpy.ndarray):
                    print(f"{key} is numpy array with shape: {value.shape}")    
                elif not isinstance(value, str) and isinstance(value, Sequence):
                    print(f"{key} is sequence with length: {len(value)}")
        
        return sample_dict

class OpPrintTypes(OpDebugBase):
    """
    Print the the type of each key 
        
    Add at the top your script to force single process:
    ```
    from fuse.utils.utils_debug import FuseDebug
    FuseDebug("debug")
    ```
    Example:
    ```
    (OpPrintTypes(first_sample_only), dict()),
    ```
    """
    def __call__(self, sample_dict: NDict) -> Any:
        if self.should_debug_sample(sample_dict):
            print(f"Sample {get_sample_id(sample_dict)} types:")
            for key in sample_dict.keypaths():
                value = sample_dict[key]
                print(f"{key} - {type(value).__name__}")

        return sample_dict