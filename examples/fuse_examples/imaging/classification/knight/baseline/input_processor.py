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
import os
import numpy as np
import skimage
import skimage.io as io
import skimage.transform as transform
import torch

import logging
import traceback
from typing import Optional, Tuple

from fuse.data.processor.processor_base import ProcessorBase

import SimpleITK as sitk

class KiTSBasicInputProcessor(ProcessorBase):
    def __init__(self,
                 input_data: str,
                 normalized_target_range: Tuple = (0, 1),
                 resize_to: Optional[Tuple] = (256, 256, 110), # 110 is the median number of slices
                 ):
        """
        Create Input processor
        :param input_data:              path to images
        :param normalized_target_range: range for image normalization
        :param resize_to:               Optional, new size of input images, keeping proportions
        """

        self.input_data = input_data
        self.normalized_target_range = normalized_target_range
        self.resize_to = resize_to

    def __call__(self,
                 inner_image_desc,
                 *args, **kwargs):
        try:
            if self.input_data.endswith('images'):
                img_path = os.path.join(self.input_data, str(inner_image_desc) + '.nii.gz')
            else:
                img_path = os.path.join(self.input_data, str(inner_image_desc), 'imaging.nii.gz')

            # read image
            data_itk = sitk.ReadImage(img_path)

            # convert to numpy
            data_npy = sitk.GetArrayFromImage(data_itk).astype(np.float32)
            
            # normalize
            #inner_image = normalize_to_range(data_npy, range=self.normalized_target_range)
            inner_image = kits_normalization(data_npy)
            
            # resize
            inner_image_height, inner_image_width, inner_image_depth = inner_image.shape[0], inner_image.shape[1], inner_image.shape[2]
            
            if inner_image_height != 512 or inner_image_width != 512:
                # there is one sample in KiTS21 with a 796 number of rows,
                # different from all the others. we disregard it for simplicity
                return None
            if self.resize_to is not None:
                h_ratio = self.resize_to[0] / inner_image_height
                w_ratio = self.resize_to[1] / inner_image_width
                if h_ratio>=1 and w_ratio>=1:
                    resize_ratio_xy = min(h_ratio, w_ratio)
                elif h_ratio<1 and w_ratio<1:
                    resize_ratio_xy = max(h_ratio, w_ratio)
                else:
                    resize_ratio_xy = 1
                #resize_ratio_z = self.resize_to[2] / inner_image_depth
                if resize_ratio_xy != 1 or inner_image_depth != self.resize_to[2]:
                    inner_image = skimage.transform.resize(inner_image,
                                                           output_shape=(int(inner_image_height * resize_ratio_xy),
                                                                         int(inner_image_width * resize_ratio_xy),
                                                                         int(self.resize_to[2])),
                                                           mode='reflect',
                                                           anti_aliasing=True
                                                           )

            image = inner_image

            # convert image from shape (H x W x D) to shape (D x H x W) 
            image = np.moveaxis(image, -1, 0)

            # add a singleton channel dimension so the image takes the shape (C x D x H x W)
            image = image[np.newaxis, :, :, :]

            # numpy to tensor
            sample = torch.from_numpy(image)

        except Exception as e:
            lgr = logging.getLogger('Fuse')
            track = traceback.format_exc()
            lgr.error(e)
            lgr.error(track)
            return None

        return sample

def kits_normalization(input_image: np.ndarray):
    # first, clip to [-62, 310] (corresponds to 0.5 and 99.5 percentile in the foreground regions)
    # then, subtract 104.9 and divide by 75.3 (corresponds to mean and std in the foreground regions, respectively)
    clip_min = -62
    clip_max = 301
    mean_val = 104.0
    std_val = 75.3
    input_image = np.minimum(np.maximum(input_image, clip_min), clip_max)
    input_image -= mean_val
    input_image /= std_val
    return input_image

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
