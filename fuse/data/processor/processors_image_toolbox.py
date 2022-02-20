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

from typing import Tuple
import pydicom
import numpy as np
import skimage
import skimage.transform as transform

class FuseProcessorsImageToolBox:
    """
    Common utils for image processors
    """

    @staticmethod
    def read_dicom_image_to_numpy(img_path: str) -> np.ndarray :
        """
        read a dicom file given a file path
        :param img_path: file path
        :return: numpy object of the dicom image
        """
        # read image
        dcm = pydicom.dcmread(img_path)
        inner_image = dcm.pixel_array
        # convert to numpy
        inner_image = np.asarray(inner_image)
        return inner_image

    @staticmethod
    def resize_image(inner_image: np.ndarray, resize_to: Tuple[int,int]) -> np.ndarray :
        """
        resize image to the required resolution
        :param inner_image: image of shape [H, W, C]
        :param resize_to: required resolution [height, width]
        :return: resized image
        """
        inner_image_height, inner_image_width = inner_image.shape[0], inner_image.shape[1]
        if inner_image_height > resize_to[0]:
            h_ratio = resize_to[0] / inner_image_height
        else:
            h_ratio = 1
        if inner_image_width > resize_to[1]:
            w_ratio = resize_to[1] / inner_image_width
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
        return inner_image

    @staticmethod
    def pad_image(inner_image: np.ndarray, padding: Tuple[float, float], resize_to: Tuple[int, int],
                         normalized_target_range: Tuple[float, float], number_of_channels: int) -> np.ndarray :
        """
        pads image to requested size ,
        pads both side equally by the same input padding size (left = right = padding[1] , up = down= padding[0] )  ,
        padding default value is zero or minimum value in normalized target range
        :param inner_image: image of shape [H, W, C]
        :param padding: required padding [x,y]
        :param resize_to: original requested resolution
        :param normalized_target_range: requested normalized image pixels range
        :param number_of_channels: number of color channels in the image
        :return: padded image
        """
        inner_image = inner_image.astype('float32')
        # "Pad" around inner image
        inner_image_height, inner_image_width = inner_image.shape[0], inner_image.shape[1]
        inner_image[0:inner_image_height, 0] = 0
        inner_image[0:inner_image_height, inner_image_width - 1] = 0
        inner_image[0, 0:inner_image_width] = 0
        inner_image[inner_image_height - 1, 0:inner_image_width] = 0

        if normalized_target_range is None:
            pad_value = 0
        else:
            pad_value = normalized_target_range[0]

        image = FuseProcessorsImageToolBox.pad_inner_image(inner_image, outer_height=resize_to[0] + 2 * padding[0],
                                                           outer_width=resize_to[1] + 2 * padding[1], pad_value=pad_value, number_of_channels=number_of_channels)
        return image

    @staticmethod
    def normalize_to_range(input_image: np.ndarray, range: Tuple[float, float] = (0, 1.0)) -> np.ndarray :
        """
        Scales tensor to range
        :param input_image: image of shape [H, W, C]
        :param range:       bounds for normalization
        :return:            normalized image
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

    def pad_inner_image(image: np.ndarray, outer_height: int, outer_width: int, pad_value: float, number_of_channels: int) -> np.ndarray :
        """
        Pastes input image in the middle of a larger one
        :param image:        image of shape [H, W, C]
        :param outer_height: final outer height
        :param outer_width:  final outer width
        :param pad_value:    value for padding around inner image
        :number_of_channels  final number of channels in the image
        :return:             padded image
        """
        inner_height, inner_width = image.shape[0], image.shape[1]
        h_offset = int((outer_height - inner_height) / 2.0)
        w_offset = int((outer_width - inner_width) / 2.0)
        if number_of_channels > 1 :
            outer_image = np.ones((outer_height, outer_width, number_of_channels), dtype=image.dtype) * pad_value
            outer_image[h_offset:h_offset + inner_height, w_offset:w_offset + inner_width, :] = image
        elif number_of_channels == 1 :
            outer_image = np.ones((outer_height, outer_width), dtype=image.dtype) * pad_value
            outer_image[h_offset:h_offset + inner_height, w_offset:w_offset + inner_width] = image
        return outer_image
