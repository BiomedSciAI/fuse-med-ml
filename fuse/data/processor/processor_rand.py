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

"""
Processor generating random ground truth - useful for testing and sanity check 
"""
from typing import Hashable, Tuple

import torch

from fuse.data.processor.processor_base import FuseProcessorBase


class FuseProcessorRandInt(FuseProcessorBase):
    def __init__(self, min: int = 0, max: int = 1, shape: Tuple = (1,)):
        self.min = min
        self.max = max
        self.shape = shape

    def __call__(self, sample_desc: Hashable):
        return {'tensor': torch.randint(self.min, self.max + 1, self.shape)}
