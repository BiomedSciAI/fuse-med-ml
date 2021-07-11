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

from typing import Union, Tuple

import numpy as np
import skimage
import torch
import cv2


class FuseUtilsImageProcessing:
    @staticmethod
    def match_img_to_input(im: np.ndarray, input: np.ndarray):
        '''
        Resize an im to the input shape
        :param im: 2D image, can be numpy or Tensor
        :param input: input image, shape [channel, W, H]. numpy or Tensor
        :return: resized im
        '''
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu().numpy()

        if len(im.shape) < 3:
            im = np.expand_dims(im, axis=0)

        if (input.shape[1], input.shape[2]) != (im.shape[1], im.shape[2]):
            resize_im = np.zeros((im.shape[0], input.shape[1], input.shape[2]))
            for c in range(im.shape[0]):
                resize_im[c, :] = cv2.resize(im[c, :], (input.shape[2], input.shape[1]),
                                             interpolation=cv2.INTER_NEAREST)
            im = resize_im

        return im

    @staticmethod
    def pad_ndimage(ndimage: np.ndarray, outer_height: int, outer_width: int, pad_value: Union[float, int]) -> Tuple[np.ndarray, int, int]:
        """
        Pastes input ndimage in the middle of a larger one
        :param ndimage:         2-dim ndimage
        :param outer_height:    final outer height
        :param outer_width:     final outer width
        :param pad_value:       value for padding around inner image
        :return:
        """
        inner_height, inner_width = ndimage.shape
        h_offset = int((outer_height - inner_height) / 2.0)
        w_offset = int((outer_width - inner_width) / 2.0)
        outer_image = np.ones((outer_height, outer_width), dtype=ndimage.dtype) * pad_value
        outer_image[h_offset:h_offset + inner_height, w_offset:w_offset + inner_width] = ndimage
        return outer_image, h_offset, w_offset

    @staticmethod
    def block_reduce_resize(img: np.ndarray, target_shape: Tuple[int, int] = (10, 5), func=np.max) -> np.ndarray:
        """
        Reduces an image by applying 'func' param on blocks, to yield a target shape.
        :param img:             2D ndarray, shape [height, width]
        :param target_shape:    target shape [height, width]
        :param func:            callable, e.g., np.max or np.average
        :return:  reduced ndimage
        """
        block_size = (int(img.shape[0] / target_shape[0]), int(img.shape[1] / target_shape[1]))
        reduced_img = skimage.measure.block_reduce(img, block_size=block_size, func=func)

        # Block reduce produces a near final result, but its dimensions might not match target_shape exactly. So resize!
        resized_reduced_img = skimage.transform.resize(reduced_img, target_shape, mode='edge', anti_aliasing=False,
                                                       anti_aliasing_sigma=None, preserve_range=True, order=0)
        return resized_reduced_img

    @staticmethod
    def preserve_range_resize(img: np.ndarray, target_shape: Tuple[int, int] = (10, 5)) -> np.ndarray:
        """
        Resizes a 2D ndarray without anti-aliasing and while preserving dtype and range
        :param img:             2D ndarray
        :param target_shape:    target shape
        :return:
        """
        resized_img = skimage.transform.resize(img, target_shape, mode='edge', anti_aliasing=False,
                                               anti_aliasing_sigma=None, preserve_range=True, order=0)
        return resized_img

    @staticmethod
    def align_ecc(img1: np.ndarray, img2: np.ndarray, num_iterations: int = 400, termination_eps: float = 1e-4,
                  transformation: int = None) -> np.ndarray:
        """
        Aligns two images using ECC estimation.
        Returns transformed img2 to match img1.
        :param img1:      ndarray of shape [height, width] and dtype=float32
        :param img2:      ndarray of shape [height, width] and dtype=float32
        :param num_iterations:  number of iterations until stopping criteria
        :param termination_eps: epsilon stopping criteria
        :param transformation: type of transformation to perform. If None (default), cv2.MOTION_AFFINE is performed.
        :return   transformed img2

        The implementation was moved to a seperate class; see FuseAlignMapECC. This function serves for backward
        compatibility
        """

        try:
            import cv2
            from fuse.utils.align.utils_align_ecc import FuseAlignMapECC

            transformation = transformation or cv2.MOTION_AFFINE

            aligner = FuseAlignMapECC(transformation_type=transformation,
                                      num_iterations=num_iterations,
                                      termination_eps=termination_eps)

            aligner.align(img1, img2)

            return aligner.img2_aligned
        except Exception as e:
            import logging
            logging.getLogger('Fuse').error("Cannot align images", e)
            return img2

    @staticmethod
    def resize_image(image: np.ndarray, resize_to: (int, int)):
        """
        Resizes image to resize_to dimensions
        :param image: ndarray of shape [height, width]
        :param resize_to: (expected_height, expected_width)
        :return: image resized to resize_to
        """

        import cv2
        image_original_h = image.shape[0]
        image_original_w = image.shape[1]

        # Resize
        if image.shape[0] > resize_to[0]:
            h_ratio = resize_to[0] / image_original_h
        else:
            h_ratio = 1
        if image.shape[1] > resize_to[1]:
            w_ratio = resize_to[1] / image_original_w
        else:
            w_ratio = 1

        resize_ratio = min(h_ratio, w_ratio)
        if resize_ratio != 1:
            return cv2.resize(image, dsize=(int(image.shape[1] * resize_ratio), int(image.shape[0] * resize_ratio)))

        return image
