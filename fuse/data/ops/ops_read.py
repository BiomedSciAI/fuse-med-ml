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

from collections.abc import Hashable
from typing import Dict, List

import h5py
import pandas as pd

from fuse.data import OpBase
from fuse.utils.file_io.file_io import read_dataframe
from fuse.utils.ndict import NDict


class OpReadDataframe(OpBase):
    """
    Op reading data from pickle file / dataframe object.
    Each row will be added as a value to sample dict.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        data_filename: str | None = None,
        columns_to_extract: List[str] | None = None,
        rename_columns: Dict[str, str] | None = None,
        key_name: str = "data.sample_id",
        key_column: str = "sample_id",
        drop_key_column: bool = True,
    ):
        """
        :param data:  input DataFrame
        :param data_filename: path to a pickled DataFrame (possible zipped)
        :param columns_to_extract: list of columns to extract from dataframe. When None (default) all columns are extracted
        :param rename_columns: rename columns from dataframe, when None (default) column names are kept
        :param key_name: name of value in sample_dict which will be used as the key/index
        :param key_column: name of the column which use as key/index. In case of None, the original dataframe index will be used to extract the values for a single sample.
        :param drop_key_column: whether to drop the key_column from the dataframe. default: True
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
            df = df.set_index(self._key_column, drop=drop_key_column)
        self._data = df.to_dict(orient="index")

    def __call__(
        self, sample_dict: NDict, prefix: str | None = None
    ) -> None | dict | List[dict]:
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


class OpReadMultiFromDataframe(OpReadDataframe):
    """
    Read multiple entries from dataframe at once.
    In that case the key expected a string that build from multiple dataframe indices separated by "@SEP@"
    For example
    df = pd.DataFrame({
        "sample_id": [0, 1, 2, 3, 4]
        "my_data": [10, 11, 12, 13, 14]
    })
    sample_dict = {
        "data.sample_id": "3@SEP@4"
    }
    will read row 3 from dataframe into sample_dict[f"my_data.0"]=13
    And  row 4 into sample_dict[f"my_data.1"]=14
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        data_filename: str | None = None,
        columns_to_extract: List[str] | None = None,
        rename_columns: Dict[str, str] | None = None,
        key_name: str = "data.sample_id",
        key_column: str = "sample_id",
        multi_key_sep: str = "@SEP@",
    ):
        super().__init__(
            data,
            data_filename,
            columns_to_extract,
            rename_columns,
            key_name,
            key_column,
        )

        self._multi_key_sep = multi_key_sep

        # convert ids to strings to support simple split and concat
        if not isinstance(next(iter(self._data.keys())), str):
            self._data = {str(k): v for k, v in self._data.items()}

    def __call__(self, sample_dict: NDict, prefix: str | None = None) -> NDict:
        multi_key = sample_dict[self._key_name]

        assert isinstance(multi_key, str), "Error: only str sample ids are supported"

        if self._multi_key_sep in multi_key:
            keys = multi_key.split(self._multi_key_sep)
        else:
            keys = [multi_key]

        for key_index, key in enumerate(keys):
            # locate the required item
            sample_data = self._data[key].copy()

            # add values tp sample_dict
            for name, value in sample_data.items():
                if prefix is None:
                    sample_dict[f"{name}.{key_index}"] = value
                else:
                    sample_dict[f"{prefix}.{name}.{key_index}"] = value

        return sample_dict


class OpReadHDF5(OpBase):
    """
    Op reading data from hd5f based dataset
    """

    def __init__(
        self,
        data_filename: str | None = None,
        columns_to_extract: List[str] | None = None,
        rename_columns: Dict[str, str] | None = None,
        key_index: str = "data.sample_id",
        key_column: str = "sample_id",
    ):
        """
        :param data_filename: path to hdf5 file
        :param columns_to_extract: list of columns to extract - dataset keys to extract. When None (default) all columns are extracted
        :param rename_columns: rename columns
        :param key_index: name of value in sample_dict which will be used as the key/index
        :param key_column: name of the column which use as key/index. In case of None, the original dataframe index will be used to extract the values for a single sample.
        """
        super().__init__()
        # store input
        self._data_filename = data_filename
        self._columns_to_extract = columns_to_extract
        self._rename_columns = rename_columns if rename_columns is not None else {}
        self._key_index = key_index
        self._key_column = key_column

        self._h5 = h5py.File(self._data_filename, "r")

        if self._columns_to_extract is None:
            self._columns_to_extract = list(self._h5.keys())

        self._num_samples = len(self._h5[self._columns_to_extract[0]])

    def num_samples(self) -> int:
        return self._num_samples

    def __call__(self, sample_dict: NDict) -> None | dict | List[dict]:
        index = sample_dict[self._key_index]
        for column in self._columns_to_extract:
            key_to_store = self._rename_columns.get(column, column)
            sample_dict[key_to_store] = self._h5[column][index]

        return sample_dict
