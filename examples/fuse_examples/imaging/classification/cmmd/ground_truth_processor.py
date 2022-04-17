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

from fuse.data.processor.processor_base import ProcessorBase


class MGGroundTruthProcessor(ProcessorBase):
    def __init__(self,
                 input_data: str):

        """
        Create Ground Truth labels - each image is attached to a binary label - either the lesion is benign or malignanat ( any type of breast cancer)
        :param input_data: path to labels
        """

        self.input_data = input_data
        input_df = pd.read_csv(input_data, header=0, names=['ID1', 'LeftRight', 'Age', 'number', 'abnormality', 'classification', 'subtype', 'file', 'view'])
        input_df.classification = np.where(input_df.classification == 'Benign', 0, 1)
        self.labels = dict(zip(input_df.file, input_df.classification))

    def __call__(self,
                 sample_desc: str,
                 *args, **kwargs):

        result = torch.tensor(self.labels[sample_desc], dtype=torch.int64)
        return result
