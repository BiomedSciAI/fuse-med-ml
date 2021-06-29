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

import torch


class FuseLossBase(torch.nn.Module):
    """
    Base class for Fuse loss functions
    """

    def __init__(self,
                 pred_name: str = None,
                 target_name: str = None,
                 weight: float = 1.0, ) -> None:
        self.pred_name = pred_name
        self.target_name = target_name
        self.weight = weight
