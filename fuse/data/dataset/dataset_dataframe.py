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

from typing import Optional, List, Dict, Union

import torch
import pandas as pd

from fuse.data.data_source.data_source_from_list import FuseDataSourceFromList
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame


class FuseDatasetDataframe(FuseDatasetDefault):
    """
    Simple dataset, based on FuseDatasetDefault, that converts dataframe into dataset.
    """
    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 data_pickle_filename: Optional[str] = None,
                 sample_desc_column: Optional[str] = 'descriptor',
                 columns_to_extract: Optional[List[str]] = None,
                 rename_columns: Optional[Dict[str, str]] = None,
                 columns_to_tensor: Optional[Union[List[str], Dict[str, torch.dtype]]] = None,
                 **kwargs):
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
        :param kwargs: additional DatasetDefault arguments. See DatasetDefault

        """
        # create processor
        processor = FuseProcessorDataFrame(data=data,
                                           data_pickle_filename=data_pickle_filename,
                                           sample_desc_column=sample_desc_column,
                                           columns_to_extract=columns_to_extract,
                                           rename_columns=rename_columns,
                                           columns_to_tensor=columns_to_tensor)

        # extract descriptor list and create datasource
        descriptors_list = processor.get_samples_descriptors()

        data_source = FuseDataSourceFromList(descriptors_list)

        super().__init__(
            data_source=data_source,
            gt_processors=None,
            input_processors=None,
            processors=processor,
            **kwargs
        )
