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
from .utils import create_knight_clinical
import json

class KiCClinicalProcessor(FuseProcessorBase):
    """
    Processor reading KiC clinical data.
    We read the json file with clinical data, convert to dataframe and read the relevant fields and then 
    follow closely the existing FuseProcessorCSV processor from fuse.data.processor.processor_csv
    """

    def __init__(self, json_filename: str, columns_to_tensor: Optional[Union[List[str], Dict[str, torch.dtype]]] = None):
        """
        Processor reading data from csv file.
        :param json_filename: path to the csv file
        :param columns_to_tensor: columns in data that should be converted into pytorch.tensor.
                        when list, all columns specified are transforms into tensors (type is decided by torch).
                        when dictionary, then each column is converted into the specified dtype.
                        When None (default) no columns are converted.
        """
        self.json_filename = json_filename
        #self.data = create_knight_clinical(json_filename)
        self.data = pd.read_json(json_filename)
        #with open(json_filename) as f:
        #    self.data = json.load(f)
        self.columns_to_tensor = columns_to_tensor
        #self.sample_desc_column = "SubjectId"
        self.sample_desc_column = 'case_id'
        self.selected_column_names = ['case_id', 'age_at_nephrectomy', 'body_mass_index', 'gender', 'comorbidities', \
                                      'smoking_history', 'radiographic_size', 'last_preop_egfr', 'aua_risk_group']

    def __call__(self, sample_desc: Hashable):
        """
        See base class
        """
        # locate the required item
        items = self.data.loc[self.data[self.sample_desc_column] == str(sample_desc)]
        # keep only selected columns
        items = items[self.selected_column_names]
        # process fields
        if items.gender.iloc[0] == 'male':
            items.gender.iloc[0] = 0
        else:
            items.gender.iloc[0] = 1
        comorbidities = 0
        for key, value in items.comorbidities.iloc[0].items():
            if value:
                comorbidities = 1
        items.comorbidities.iloc[0] = comorbidities
        if items.smoking_history.iloc[0] == 'never_smoked': 
            items.smoking_history.iloc[0] = 0
        elif items.smoking_history.iloc[0] == 'previous_smoker':
            items.smoking_history.iloc[0] = 1
        elif items.smoking_history.iloc[0] == 'current_smoker':
            items.smoking_history.iloc[0] = 2

        if items.last_preop_egfr.iloc[0]['value'] == '>=90':
            items.last_preop_egfr.iloc[0] = 90
        else:
            items.last_preop_egfr.iloc[0] = items.last_preop_egfr.iloc[0]['value']

        # Task 1 labels:
        if items.aua_risk_group.iloc[0] in ['high_risk', 'very_high_risk']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
            items['task_1_label'] = 1 # CanAT
        else:
            items['task_1_label'] = 0 # NoAT

        # Task 2 labels:
        if items.aua_risk_group.iloc[0] == 'benign':
            items['task_2_label'] = 0 
        elif items.aua_risk_group.iloc[0] == 'low_risk':
            items['task_2_label'] = 1
        elif items.aua_risk_group.iloc[0] == 'intermediate_risk':
            items['task_2_label'] = 2
        elif items.aua_risk_group.iloc[0] == 'high_risk':
            items['task_2_label'] = 3
        elif items.aua_risk_group.iloc[0] == 'very_high_risk':
            items['task_2_label'] = 4
        else:
            ValueError('Wrong risk class')
        # convert to dictionary - assumes there is only one item with the requested descriptor
        sample_data = items.to_dict('records')[0]
        for key in sample_data.keys():
            if 'output' in key:
                if isinstance(sample_data[key], str):
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
        else:
            if isinstance(sample[key], Tensor):
                sample[key] = sample[key]
            else:
                sample[key] = torch.tensor(sample[key], dtype=tensor_dtype)
