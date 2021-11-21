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

import ast
import pandas as pd

from fuse.data.processor.processor_base import FuseProcessorBase
import logging
from typing import Hashable, List, Optional, Dict, Union
from torch import Tensor
import torch

class FuseProcessorCSV(FuseProcessorBase):
    """
    Processor reading data from csv file.
    Covert each row to a dictionary
    """

    def __init__(self, csv_filename: str, sample_desc_column: str='descriptor', columns_to_tensor: Optional[Union[List[str], Dict[str, torch.dtype]]] = None):
        """
        Processor reading data from csv file.
        :param csv_filename: path to the csv file
        :param sample_desc_column: name of the sample descriptor column within the csv file
        :param columns_to_tensor: columns in data that should be converted into pytorch.tensor.
                        when list, all columns specified are transforms into tensors (type is decided by torch).
                        when dictionary, then each column is converted into the specified dtype.
                        When None (default) no columns are converted.
        """
        self.sample_desc_column = sample_desc_column
        self.csv_filename = csv_filename
        # read csv
        self.data = pd.read_csv(csv_filename)
        self.columns_to_tensor = columns_to_tensor

    def __call__(self, sample_desc: Hashable):
        """
        See base class
        """
        # locate the required item
        items = self.data.loc[self.data[self.sample_desc_column] == str(sample_desc)]
        # convert to dictionary - assumes there is only one item with the requested descriptor
        sample_data = items.to_dict('records')[0]
        for key in sample_data.keys():
            if 'output' in key and isinstance(sample_data[key], str):
                tuple_data = sample_data[key]
                if tuple_data.startswith('[') and tuple_data.endswith(']'):
                    sample_data[key] = ast.literal_eval(tuple_data.replace(" ", ","))
        # convert to tensor
        if self.columns_to_tensor is not None:
            if isinstance(self.columns_to_tensor, list):
                for col in self.columns_to_tensor:
                    self.convert_to_tensor(sample_data, col)
            elif isinstance(self.columns_to_tensor, dict):
                for col, tensor_dtype in self.columns_to_tensor.items():
                    self.convert_to_tensor(sample_data, col, tensor_dtype)
        return sample_data

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
        elif isinstance(sample[key], Tensor):
            sample[key] = sample[key]
        else:
            sample[key] = torch.tensor(sample[key], dtype=tensor_dtype)
