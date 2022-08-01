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

from fuse.data.processor.processor_base import ProcessorBase
import logging
from typing import Hashable, List, Optional, Dict, Union
from torch import Tensor
import torch


class KiCClinicalProcessor(ProcessorBase):
    """
    Processor reading KiC clinical data.
    """

    def __init__(
        self, json_filename: str, columns_to_tensor: Optional[Union[List[str], Dict[str, torch.dtype]]] = None
    ):
        """
        Processor reading data from csv file.
        :param json_filename: path to the csv file
        :param columns_to_tensor: columns in data that should be converted into pytorch.tensor.
                        when list, all columns specified are transforms into tensors (type is decided by torch).
                        when dictionary, then each column is converted into the specified dtype.
                        When None (default) no columns are converted.
        """
        self.json_filename = json_filename
        self.data = pd.read_json(json_filename)
        self.columns_to_tensor = columns_to_tensor
        self.sample_desc_column = "case_id"
        self.selected_column_names = [
            "case_id",
            "age_at_nephrectomy",
            "body_mass_index",
            "gender",
            "comorbidities",
            "smoking_history",
            "radiographic_size",
            "last_preop_egfr",
        ]

        # extract only specified columns (in case not specified, extract all)
        self.data = self.data[self.selected_column_names]

        # convert to dictionary: {index -> {column -> value}}
        self.data = self.data.set_index(self.sample_desc_column)
        self.data = self.data.to_dict(orient="index")

    def __call__(self, sample_desc: Hashable):
        """
        See base class
        """
        # locate the required item
        sample_data = self.data[sample_desc].copy()
        # process fields
        if sample_data["gender"].lower() == "male":
            sample_data["gender"] = 0
        else:
            sample_data["gender"] = 1
        comorbidities = 0
        for key, value in sample_data["comorbidities"].items():
            if value:
                comorbidities = 1
        sample_data["comorbidities"] = comorbidities
        if sample_data["smoking_history"] == "never_smoked":
            sample_data["smoking_history"] = 0
        elif sample_data["smoking_history"] == "previous_smoker":
            sample_data["smoking_history"] = 1
        elif sample_data["smoking_history"] == "current_smoker":
            sample_data["smoking_history"] = 2

        if sample_data["last_preop_egfr"] is None:
            sample_data["last_preop_egfr"] = 77  # this is the median value on the training set
        else:
            if sample_data["last_preop_egfr"]["value"] is None:
                sample_data["last_preop_egfr"] = 77  # this is the median value on the training set
            elif sample_data["last_preop_egfr"]["value"] in (">=90", ">90"):
                sample_data["last_preop_egfr"] = 90
            else:
                sample_data["last_preop_egfr"] = sample_data["last_preop_egfr"]["value"]

        if sample_data["radiographic_size"] is None:
            sample_data["radiographic_size"] = 4.1  # this is the median value on the training set

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
            lgr = logging.getLogger("Fuse")
            lgr.error(f"Column {key} does not exit in dataframe, it is ignored and not converted to {tensor_dtype}")
        else:
            if isinstance(sample[key], Tensor):
                sample[key] = sample[key]
            else:
                sample[key] = torch.tensor(sample[key], dtype=tensor_dtype)


class KiCGTProcessor(ProcessorBase):
    """
    Processor reading KiC ground truth data.
    """

    def __init__(
        self,
        json_filename: str,
        columns_to_tensor: Optional[Union[List[str], Dict[str, torch.dtype]]] = None,
        test_labels=False,
    ):
        """
        Processor reading data from csv file.
        :param json_filename: path to the csv file
        :param columns_to_tensor: columns in data that should be converted into pytorch.tensor.
                        when list, all columns specified are transforms into tensors (type is decided by torch).
                        when dictionary, then each column is converted into the specified dtype.
                        When None (default) no columns are converted.
        """
        self.json_filename = json_filename
        if not test_labels:
            self.data = pd.read_json(json_filename)
        else:
            # handle test label file format:
            label_data = pd.read_json(json_filename, typ="series")
            label_data = label_data.to_dict()
            self.data = pd.DataFrame(columns=["case_id", "aua_risk_group"])
            self.data["case_id"] = label_data.keys()
            self.data["aua_risk_group"] = label_data.values()
        self.columns_to_tensor = columns_to_tensor
        self.sample_desc_column = "case_id"
        self.selected_column_names = ["case_id", "aua_risk_group"]
        # extract only specified columns (in case not specified, extract all)
        self.data = self.data[self.selected_column_names]
        # convert to dictionary: {index -> {column -> value}}
        self.data = self.data.set_index(self.sample_desc_column)
        self.data = self.data.to_dict(orient="index")

    def __call__(self, sample_desc: Hashable):
        """
        See base class
        """
        # locate the required item
        sample_data = self.data[sample_desc].copy()
        # process fields
        # Task 1 labels:
        if sample_data["aua_risk_group"] in ["high_risk", "very_high_risk"]:  # 1:'3','4'  0:'0','1a','1b','2a','2b'
            sample_data["task_1_label"] = 1  # CanAT
        else:
            sample_data["task_1_label"] = 0  # NoAT

        # Task 2 labels:
        if sample_data["aua_risk_group"] == "benign":
            sample_data["task_2_label"] = 0
        elif sample_data["aua_risk_group"] == "low_risk":
            sample_data["task_2_label"] = 1
        elif sample_data["aua_risk_group"] == "intermediate_risk":
            sample_data["task_2_label"] = 2
        elif sample_data["aua_risk_group"] == "high_risk":
            sample_data["task_2_label"] = 3
        elif sample_data["aua_risk_group"] == "very_high_risk":
            sample_data["task_2_label"] = 4
        else:
            ValueError("Wrong risk class")

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
            lgr = logging.getLogger("Fuse")
            lgr.error(f"Column {key} does not exit in dataframe, it is ignored and not converted to {tensor_dtype}")
        else:
            if isinstance(sample[key], Tensor):
                sample[key] = sample[key]
            else:
                sample[key] = torch.tensor(sample[key], dtype=tensor_dtype)
