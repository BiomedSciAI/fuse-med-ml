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

from typing import Dict, List, Sequence

from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.managers.callbacks.callback_base import FuseCallback


class FuseAugmentorBatchCallback(FuseCallback):
    """
    Simple class which gets augmentation pipeline and apply augmentation on a batch level batch dict
    """
    def __init__(self, aug_pipeline: List, modes: Sequence[str] = ('train',)):
        """
        :param aug_pipeline: See  FuseAugmentorDefault
        :param modes: modees to apply the augmentation: 'train', 'validation' and/or 'infer'
        """
        self._augmentor = FuseAugmentorDefault(aug_pipeline)
        self._modes = modes

    def on_data_fetch_end(self, mode: str, batch: int, batch_dict: Dict = None) -> None:
        if mode in self._modes:
            self._augmentor(batch_dict)