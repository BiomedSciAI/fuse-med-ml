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
import logging
import torch
import pandas as pd
from torch import Tensor

from fuse.data.processor.processor_base import FuseProcessorBase


class FuseProcessorDataFrame(FuseProcessorBase):
    """
    Processor reading data from pickle file / dataframe object.
    Covert each row to a dictionary
    """

    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 data_pickle_filename: Optional[str] = None,
                 sample_desc_column: Optional[str] = 'descriptor',
                 columns_to_extract: Optional[List[str]] = None,
                 rename_columns: Optional[Dict[str, str]] = None,
                 columns_to_tensor: Optional[Union[List[str], Dict[str, torch.dtype]]] = None):
        """
        :param data:  input DataFrame
        :param data_pickle_filename: path to a pickled DataFrame (possible gzipped)
        :param sample_desc_column: name of the sample descriptor column within the pickle file,
                                   if set to None.will simply use dataframe index as descriptors
        :param columns_to_extract: list of columns to extract from dataframe. When None (default) all columns are extracted
        :param rename_columns: rename columns from dataframe, when None (default) column names are kept
        :param columns_to_tensor: columns in data that should be converted into pytorch.tensor.
                        when list, all columns specified are transforms into tensors (type is decided by torch).
                        when dictionary, then each column is converted into the specified dtype.
                        When None (default) no columns are converted.
        """
        # verify input
        lgr = logging.getLogger('Fuse')
        if data is None and data_pickle_filename is None:
            msg = "Error in FuseProcessorDataFrame - need to provide either in-memory DataFrame or a path to pickled DataFrame."
            lgr.error(msg)
            raise Exception(msg)
        elif data is not None and data_pickle_filename is not None:
            msg = "Error in FuseProcessorDataFrame - need to provide either 'data' or 'data_pickle_filename' args, bot not both."
            lgr.error(msg)
            raise Exception(msg)

        # read dataframe
        if data is not None:
            self.data = data
            self.pickle_filename = 'in-memory'
        elif data_pickle_filename is not None:
            self.data = pd.read_pickle(data_pickle_filename)
            self.pickle_filename = data_pickle_filename

        # store input arguments
        self.sample_desc_column = sample_desc_column
        self.columns_to_extract = columns_to_extract
        self.columns_to_tensor = columns_to_tensor

        # extract only specified columns (in case not specified, extract all)
        if self.columns_to_extract is not None:
            self.data = self.data[self.columns_to_extract]

        # rename columns
        if rename_columns is not None:
            self.data.rename(rename_columns, axis=1, inplace=True)

        # convert to dictionary: {index -> {column -> value}}
        self.data = self.data.set_index(self.sample_desc_column)
        self.data = self.data.to_dict(orient='index')

    def __call__(self, sample_desc: Hashable):
        """
        See base class
        """
        # locate the required item
        sample_data = self.data[sample_desc].copy()

        # convert to tensor
        if self.columns_to_tensor is not None:
            if isinstance(self.columns_to_tensor, list):
                for col in self.columns_to_tensor:
                    self.convert_to_tensor(sample_data, col)
            elif isinstance(self.columns_to_tensor, dict):
                for col, tensor_dtype in self.columns_to_tensor.items():
                    self.convert_to_tensor(sample_data, col, tensor_dtype)

        return sample_data

    def get_samples_descriptors(self) -> List[Hashable]:
        """
        :return: list of descriptors dataframe index values
        """
        return list(self.data.keys())

    @staticmethod
    def convert_to_tensor(sample: dict, key: str, tensor_dtype: Optional[str] = None) -> None:
        """
        Convert value to tensor, use tensor_dtype to specify non-default type/
        :param sample: sample dictionary
        :param key: key of item in sample dict to convert
        :param tensor_dtype: Optional, None for default,.
        """
        if key not in sample:
            lgr = logging.getLogger('Fuse')
            lgr.error(f'Column {key} does not exit in dataframe, it is ignored and not converted to {tensor_dtype}')
        else:
            if isinstance(sample[key], Tensor):
                sample[key] = sample[key]
            else:
                sample[key] = torch.tensor(sample[key], dtype=tensor_dtype)
