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
import skimage
import skimage.io as io
import skimage.transform as transform
import torch

import logging
import traceback
from typing import Optional, Tuple

from fuse.data.processor.processor_base import FuseProcessorBase


class FuseSkinInputProcessor(FuseProcessorBase):
    def __init__(self,
                 input_data: str,
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

        self.input_data = input_data
        self.normalized_target_range = normalized_target_range
        self.resize_to = np.subtract(resize_to, (2*padding[0], 2*padding[1]))
        self.padding = padding

    def __call__(self,
                 inner_image_desc,
                 *args, **kwargs):
        try:
            img_path = self.input_data + str(inner_image_desc) + '.jpg'

            # read image
            inner_image = skimage.io.imread(img_path)

            # convert to numpy
            inner_image = np.asarray(inner_image)
            
            # normalize
            inner_image = normalize_to_range(inner_image, range=self.normalized_target_range)

            # resize
            inner_image_height, inner_image_width = inner_image.shape[0], inner_image.shape[1]
            
            if self.resize_to is not None:
                if inner_image_height > self.resize_to[0]:
                    h_ratio = self.resize_to[0] / inner_image_height
                else:
                    h_ratio = 1
                if inner_image_width > self.resize_to[1]:
                    w_ratio = self.resize_to[1] / inner_image_width
                else:
                    w_ratio = 1

                resize_ratio = min(h_ratio, w_ratio)
                if resize_ratio != 1:
                    inner_image = skimage.transform.resize(inner_image,
                                                           output_shape=(int(inner_image_height * resize_ratio),
                                                                         int(inner_image_width * resize_ratio)),
                                                           mode='reflect',
                                                           anti_aliasing=True
                                                           )

            # padding
            if self.padding is not None:
                # "Pad" around inner image
                inner_image = inner_image.astype('float32')

                inner_image_height, inner_image_width = inner_image.shape[0], inner_image.shape[1]
                inner_image[0:inner_image_height, 0] = 0
                inner_image[0:inner_image_height, inner_image_width-1] = 0
                inner_image[0, 0:inner_image_width] = 0
                inner_image[inner_image_height-1, 0:inner_image_width] = 0

                if self.normalized_target_range is None:
                    pad_value = 0
                else:
                    pad_value = self.normalized_target_range[0]

                image = pad_image(inner_image, outer_height=self.resize_to[0] + 2*self.padding[0], outer_width=self.resize_to[1] + 2*self.padding[1], pad_value=pad_value)

            else:
                image = inner_image

            # convert image from shape (H x W x C) to shape (C x H x W) with C=3
            image = np.moveaxis(image, -1, 0)

            # numpy to tensor
            sample = torch.from_numpy(image)

        except Exception as e:
            lgr = logging.getLogger('Fuse')
            track = traceback.format_exc()
            lgr.error(e)
            lgr.error(track)
            return None

        return sample


def normalize_to_range(input_image: np.ndarray, range: Tuple = (-1.0, 1.0)):
    """
    Scales tensor to range
    @param input_image: image of shape (H x W x C)
    @param range:       bounds for normalization
    @return:            normalized image
    """
    max_val = input_image.max()
    min_val = input_image.min()
    if min_val == max_val == 0:
        return input_image
    input_image = input_image - min_val
    input_image = input_image / (max_val - min_val)
    input_image = input_image * (range[1] - range[0])
    input_image = input_image + range[0]
    return input_image


def pad_image(image: np.ndarray, outer_height: int, outer_width: int, pad_value: Tuple):
    """
    Pastes input image in the middle of a larger one
    @param image:        image of shape (H x W x C)
    @param outer_height: final outer height
    @param outer_width:  final outer width
    @param pad_value:    value for padding around inner image
    @return:             padded image
    """
    inner_height, inner_width = image.shape[0], image.shape[1]
    h_offset = int((outer_height - inner_height) / 2.0)
    w_offset = int((outer_width - inner_width) / 2.0)
    outer_image = np.ones((outer_height, outer_width, 3), dtype=image.dtype) * pad_value
    outer_image[h_offset:h_offset + inner_height, w_offset:w_offset + inner_width, :] = image

    return outer_image
