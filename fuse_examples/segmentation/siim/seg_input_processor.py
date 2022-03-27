
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

import numpy as np
from skimage.io import imread
import torch

from typing import Optional, Tuple

from fuse.data.processor.processor_base import FuseProcessorBase


class SegInputProcessor(FuseProcessorBase):
    def __init__(self,
                 input_data: str = None,
                 name: str = 'image', # can be 'image' or 'mask'
                 normalized_target_range: Tuple = (-1, 1),
                 resize_to: Optional[Tuple] = (299, 299),
                 padding: Optional[Tuple] = (0, 0),
                 ):
        """
        Create Input processor
        :param input_data:              path to images
        :param normalized_target_range: range for image normalization
        :param resize_to:               Optional, new size of input images, keeping proportions
        :param padding:                 Optional, padding size
        """
        self.name = name
        if self.name == 'image':
            self.im_inx = 0
        elif self.name == 'mask':
            self.im_inx = 1
        else:
            print('Wrong input!!')

    def __call__(self,
                 image_fn,
                 *args, **kwargs):

        try:
            image_fn = image_fn[self.im_inx] 
            image = imread(image_fn)

            if self.name == 'image':
                image = image.astype('float32')
                image = image / 255.0
            else:
                image = image > 0
                image = image.astype('float32')

            # convert image from shape (H x W x C) to shape (C x H x W) with C=3
            if len(image.shape) > 2:
                image = np.moveaxis(image, -1, 0)
            else:
                image = np.expand_dims(image, 0)

            # numpy to tensor
            sample = torch.from_numpy(image)

        except:
            return None

        return sample
