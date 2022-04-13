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
                 year: Optional[str] = '2016'):

        """
        Create Ground Truth labels
        :param input_data: path to labels
        :param train:      Optional, specify if we are in training phase
        :param year:       Optional, ISIC challenge year
        """

        self.input_data = input_data

        if year == '2016':
            input_df = pd.read_csv(input_data, names=['id', 'label'])
            if train:
                input_df.label = np.where(input_df.label == 'benign', 0, 1)
        else:  # year = 2017
            input_df = pd.read_csv(input_data, header=0, names=['id', 'label', 'other'])
        self.labels = dict(zip(input_df.id, input_df.label))

    def __call__(self,
                 sample_desc,
                 *args, **kwargs):

        result = {'tensor': torch.tensor(self.labels[sample_desc], dtype=torch.int64)}
        return result
