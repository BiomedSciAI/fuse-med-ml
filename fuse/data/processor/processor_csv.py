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
from typing import Hashable

import pandas as pd

from fuse.data.processor.processor_base import FuseProcessorBase


class FuseProcessorCSV(FuseProcessorBase):
    """
    Processor reading data from csv file.
    Covert each row to a dictionary
    """

    def __init__(self, csv_filename: str, sample_desc_column='descriptor'):
        """
        Processor reading data from csv file.
        :param csv_filename: path to the csv file
        :param sample_desc_column: name of the sample descriptor column within the csv file
        """
        self.sample_desc_column = sample_desc_column
        self.csv_filename = csv_filename
        # read csv
        self.data = pd.read_csv(csv_filename)

    def __call__(self, sample_desc: Hashable):
        """
        See base class
        """
        # locate the required item
        items = self.data.loc[self.data[self.sample_desc_column] == str(sample_desc)]
        # convert to dictionary - assumes there is only one item with the requested descriptor
        sample_data = items.to_dict('records')[0]
        for key in sample_data.keys():
            if 'output' in key:
                if isinstance(sample_data[key], str):
                    tuple_data = sample_data[key]
                    if tuple_data.startswith('[') and tuple_data.endswith(']'):
                        sample_data[key] = ast.literal_eval(tuple_data.replace(" ", ","))

        return sample_data
