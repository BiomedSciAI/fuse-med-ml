"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

"""
Fuse miscellaneous utils
"""
import logging
import sys
import time
from collections import Iterable
from threading import Lock
from typing import List, Union, Sequence, Optional, Hashable, Any
import numpy as np

import pandas as pd
from torch import Tensor

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseUtilsMisc:
    @classmethod
    def flatten(cls, l: List[List]) -> List:
        """
        Flatten a list of lists, yielding a 1-d list of strings
        :param l: list of lists, possible irregular shapes
        :return: generator of single str elements
        """
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from cls.flatten(el)
            else:
                yield el

    @staticmethod
    def query_yes_no(question: str, default: Optional[str] = None) -> bool:
        """
        Ask the user yes/no question
        :param question: string question
        :param default: default answer in case the user type enter. if set to none the question will be asked again.
        :return: the answer
        """
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            sys.stdout.flush()
            choice = sys.stdin.readline().rstrip('\r\n').lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")

    @staticmethod
    def squeeze_obj(obj: Any) -> Any:
        """
        Get batch with single sample tenosor / numpy / list and squeeze the batch dimension
        :param obj: the object to sqeeze
        :return: squeezed object
        """
        if isinstance(obj, Tensor):
            obj_tensor: Tensor = obj
            if obj_tensor.shape[0] == 1:
                obj_tensor = obj_tensor.squeeze(dim=0)
                return obj_tensor
        elif isinstance(obj, np.ndarray):
            obj_np: np.ndarray = obj
            if obj_np.shape[0] == 1:
                obj_np = obj_np.squeeze(axis=0)
                return obj_np
        elif isinstance(obj, list):
            obj_lst: list = obj
            if len(obj_lst) == 1:
                return obj_lst[0]

        return obj

    @staticmethod
    def batch_dict_to_string(batch_dict: dict) -> str:
        """
        Convert batch dict to string, including the keys, types and shapes.
        :param batch_dict: might be any dict
        :return: string representation of the batch dict
        """
        res = ''
        all_keys = FuseUtilsHierarchicalDict.get_all_keys(batch_dict)
        for key in all_keys:
            value = FuseUtilsHierarchicalDict.get(batch_dict, key)
            res += f'{key} : type={type(value)}'
            if isinstance(value, (np.ndarray, Tensor)):
                res += f', dtype={value.dtype}, shape={value.shape}'
            elif isinstance(value, Sequence):
                res += f', length={len(value)}'
            res += '\n'
        return res


def time_display(seconds, granularity=3):
    intervals = (
        # ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )
    if seconds < 1:
        return "< second"
    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("%d %s" % (value, name))
    return ', '.join(result[:granularity])


def get_pretty_dataframe(df, col_width=25):
    # check is col_width needs to be widen (so that dashes are in one line)
    max_val_width = np.vectorize(len)(df.values.astype(str)).max() # get maximum length of all values
    max_col_width = max([len(x) for x in df.columns])  # get the maximum lengths of all column names
    col_width = max(max_col_width, max_val_width, col_width)

    dashes = (col_width + 2) * len(df.columns.values)
    df_as_string = f"\n{'-' * dashes}\n"
    for col in df.columns.values:
        df_as_string += f'| {col:{col_width}}'
    df_as_string += f"|\n{'-' * dashes}\n"
    for idx, row in df.iterrows():
        for col in df.columns.values:
            df_as_string += f'| {row[col]:{col_width}}'

        df_as_string += f"|\n{'-' * dashes}\n"
    return df_as_string


def get_time_delta(begin_time: float) -> str:
    """
    Returns a string representing the difference in time between now and the begin_time

    :param begin_time: start
    :return: diff between now and begin_time
    """
    time_as_float = time.time() - begin_time
    return time_display(time_as_float)


def autodetect_input_source(input_source: Union[str, pd.DataFrame, Sequence[Hashable]] = None):
    """
    Loads sample descriptors from multiple auto-detectable possible sources:
    1. DataFrame (instance or path to pickled object)
    2. Python list of sample descriptors
    3. Text file (needs to end with '.txt' or '.text' extension)
    :param input_source:
    :return:
    """

    df = None

    # DataFrame instance
    # ------------------
    if isinstance(input_source, pd.DataFrame):
        df = input_source

    # Python list of sample descriptors
    # ---------------------------------
    elif isinstance(input_source, (list, tuple)):
        df = pd.DataFrame({'sample_desc': input_source})

    # Path to:
    # --------
    elif isinstance(input_source, str):
        file_extension = input_source.lower().split('.')[-1]

        # Text file
        # ---------
        if file_extension in ['text', 'txt']:
            with open(input_source, 'r') as f:
                lines = f.readlines()
                sample_descs = [line.strip() for line in lines]
                df = pd.DataFrame({'sample_desc': sample_descs})

        # Pickled DataFrame
        # -----------------
        if file_extension in ['p', 'pkl', 'pickle', 'gz', 'bz2', 'zip', 'xz']:
            df = pd.read_pickle(input_source)

    return df


class Singleton(type):
    """
    Singleton metaclass
    """
    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            else:
                if len(args) + len(kwargs) > 0:
                    logging.getLogger('Fuse').warning('Ignoring a redefinition of the singleton of class {class_.__name__}')

            return cls._instances[cls]
