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

from typing import Dict, Any, Optional, List

import torch

from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
import numpy as np


def post_processing(batch_dict: Dict,
                ) -> None:


    label_tensor = torch.tensor(FuseUtilsHierarchicalDict.get(batch_dict, 'data.ClinSig')+0,dtype=torch.int64)
    FuseUtilsHierarchicalDict.set(batch_dict, 'data.ground_truth', label_tensor)




    zone = FuseUtilsHierarchicalDict.get(batch_dict, 'data.zone')
    zone2feature = {
        'PZ': torch.tensor(np.array([0, 0, 0]), dtype=torch.float32),
        'TZ': torch.tensor(np.array([0, 0, 1]), dtype=torch.float32),
        'AS': torch.tensor(np.array([0, 1, 0]), dtype=torch.float32),
        'SV': torch.tensor(np.array([1, 0, 0]), dtype=torch.float32),
    }
    FuseUtilsHierarchicalDict.set(batch_dict, 'data.tensor_clinical', zone2feature[zone])
