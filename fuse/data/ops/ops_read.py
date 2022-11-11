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

from typing import Hashable, List, Optional, Dict, Union
from fuse.utils.file_io.file_io import read_dataframe
import pandas as pd

from fuse.data import OpBase
from fuse.utils.ndict import NDict


class OpReadDataframe(OpBase):
    """
    Op reading data from pickle file / dataframe object.
    Each row will be added as a value to sample dict.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_filename: Optional[str] = None,
        columns_to_extract: Optional[List[str]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        key_name: str = "data.sample_id",
        key_column: str = "sample_id",
    ):
        """
        :param data:  input DataFrame
        :param data_filename: path to a pickled DataFrame (possible zipped)
        :param columns_to_extract: list of columns to extract from dataframe. When None (default) all columns are extracted
        :param rename_columns: rename columns from dataframe, when None (default) column names are kept
        :param key_name: name of value in sample_dict which will be used as the key/index
        :param key_column: name of the column which use as key/index
        """
        super().__init__()

        # store input
        self._data_filename = data_filename
        self._columns_to_extract = columns_to_extract
        self._rename_columns = rename_columns
        self._key_name = key_name
        self._key_column = key_column
        df = data

        # verify input
        if data is None and data_filename is None:
            msg = "Error: need to provide either in-memory DataFrame or a path to file."
            raise Exception(msg)
        elif data is not None and data_filename is not None:
            msg = "Error: need to provide either 'data' or 'data_filename' args, bot not both."
            raise Exception(msg)

        # read dataframe
        if self._data_filename is not None:
            df = read_dataframe(self._data_filename)

        # extract only specified columns (in case not specified, extract all)
        if self._columns_to_extract is not None:
            df = df[self._columns_to_extract]

        # rename columns
        if self._rename_columns is not None:
            df = df.rename(self._rename_columns, axis=1)

        # convert to dictionary: {index -> {column -> value}}
        if self._key_column is not None:
            df = df.set_index(self._key_column)
        self._data = df.to_dict(orient="index")

    def __call__(self, sample_dict: NDict, prefix: Optional[str] = None) -> Union[None, dict, List[dict]]:
        """
        See base class

        :param prefix: specify a prefix for the sample dict keys.
                       For example, with prefix 'data.features' and a df with the columns ['height', 'weight', 'sex'],
                       the matching keys will be: 'data.features.height', 'data.features.weight', 'data.features.sex'.
        """
        key = sample_dict[self._key_name]

        # locate the required item
        sample_data = self._data[key].copy()

        # add values tp sample_dict
        for name, value in sample_data.items():
            if prefix is None:
                sample_dict[name] = value
            else:
                sample_dict[f"{prefix}.{name}"] = value

        return sample_dict

    def get_all_keys(self) -> List[Hashable]:
        """
        :return: list of  dataframe index values
        """
        return list(self.data.keys())
