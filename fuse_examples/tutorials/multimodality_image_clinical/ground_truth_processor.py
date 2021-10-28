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

from typing import Optional

import torch
import pandas as pd
import numpy as np

from fuse.data.processor.processor_base import FuseProcessorBase


class FuseSkinGroundTruthProcessor(FuseProcessorBase):
    def __init__(self,
                 input_data: str,
                 train: Optional[bool] = True,
                 year: Optional[str] = '2019'):

        """
        Create Ground Truth labels
        :param input_data: path to labels
        :param train:      Optional, specify if we are in training phase
        :param year:       Optional, ISIC challenge year
        """
        self.input_data = input_data
        self._year = year

        # 2019
        input_df = pd.read_csv(input_data, header=0)
        input_df = input_df.set_index("image")
        self.labels = input_df.to_dict(orient="index")
        self._num_classes = 9
        self._class_index = {
            'MEL': 0,
            'NV': 1,
            'BCC': 2,
            'AK': 3,
            'BKL': 4,
            'DF': 5,
            'VASC': 6,
            'SCC': 7,
            'UNK': 8,
        }
        

    def __call__(self,
                 sample_desc,
                 *args, **kwargs):

        labels = self.labels[sample_desc]
        result = torch.zeros(self._num_classes, dtype=torch.int64)
        for class_name, class_value in labels.items():
            result[self._class_index[class_name]] = class_value
        assert result.sum() == 1
        
        result = {'tensor': result.argmax()}
        return result
