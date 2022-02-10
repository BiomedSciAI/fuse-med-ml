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
import os
from skimage import measure
import torch
from typing import Optional, Tuple

from fuse.data.processor.processor_base import FuseProcessorBase
from fuse.data.processor.processors_image_toolbox import FuseProcessorsImageToolBox


class FuseMGInputProcessor(FuseProcessorBase):
    """
    This processor expects configuration parameters to process an image and path to a mammography dicom file
    it then reads it (just the image),
    it then do the pre-processing steps:
    1. standardize the image to be focused on the breast and move the breast to the left and noramlized the values to requested range
    2. resize the image to requested size
    3. pad the image according to requested
    returns the processed image as torch tensor object
    """
    def __init__(self,
                 input_data: str,
                 normalized_target_range: Tuple = (0, 1),
                 resize_to: Optional[Tuple] = (2200, 1200),
                 padding: Optional[Tuple] = (60, 60),
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
        self.resize_to = np.subtract(resize_to, (2 * padding[0], 2 * padding[1]))
        self.padding = padding

    def __call__(self,
                 inner_image_desc,
                 *args, **kwargs):

        image_full_path = self.input_data + inner_image_desc
        image_full_path = os.path.join(self.input_data, inner_image_desc)
        inner_image = FuseProcessorsImageToolBox.read_dicom_image_to_numpy(image_full_path)

        inner_image = standardize_breast_image(inner_image, self.normalized_target_range)

        # resize
        if self.resize_to is not None:
            inner_image = FuseProcessorsImageToolBox.resize_image(inner_image, self.resize_to)

        # padding
        if self.padding is not None:
            image = FuseProcessorsImageToolBox.pad_image(inner_image, self.padding, self.resize_to, self.normalized_target_range, 1)

        else:
            image = inner_image

        # numpy to tensor
        sample = torch.from_numpy(image)
        sample = torch.unsqueeze(sample, dim=0)

        return sample


def find_breast_aabb(img : np.ndarray, dark_area_threshold : int = 10, blocks_num : int = 30) -> Tuple[int ,int, int, int]:
    """
    split the images into blocks, each block containing (1/30 x 1/30) of the image.
    All blocks above a threshold (10) are considered non-empty.
    Then, the biggest connected component (at the blocks level) is extracted, and its axis-aligned bbox is returned.
    :param img: Image instance , expected 2d breast integer grayscale image where 0 is black background color
    :param dark_area_threshold: defines grayscale level from which lower is consider dark / outside of body
    :param blocks_num: number of blocks the algorithm split the images into
    :return: four coordinates
    """
    bl_rows = img.shape[0]//blocks_num
    bl_cols = img.shape[1]//blocks_num

    rows_starts = list(range(blocks_num))
    cols_starts = list(range(blocks_num))

    rows_starts = list(map(lambda x:x*bl_rows, rows_starts))
    cols_starts = list(map(lambda x: x * bl_cols, cols_starts))

    cells = np.zeros((blocks_num,blocks_num))

    for ri,r in enumerate(rows_starts):
        for ci,c in enumerate(cols_starts):
            r_end = min(img.shape[0]-1,r+bl_rows)
            c_end = min(img.shape[1] - 1, c + bl_cols)
            cells[ri,ci] = np.mean(img[r:r_end,c:c_end])

    cells_binary = np.zeros((blocks_num,blocks_num),dtype=bool)
    cells_binary[cells>dark_area_threshold] = True #TODO: this is a hardcoded threshold, see if there's a better alternative
    blobs_labels = measure.label(cells_binary, background=0)
    regions = measure.regionprops(blobs_labels)

    regions_areas = [r.area for r in regions]
    if len(regions)<1:
        # expected to have at least background and one object. TODO: consider reporting error and returning AABB of the full image.
        print('Warning: could not crop properly! fallbacking to full image')
        return 0,0,img.shape[0]-1,img.shape[1]-1


    for i,r in enumerate(regions):
        if r.label==0:
            regions_areas[i]=-1
            break
    max_ind = np.argmax(regions_areas)
    bbox = regions[max_ind].bbox

    full_img_bbox = [bbox[0]*bl_rows,bbox[1]*bl_cols,(bbox[2]+1)*bl_rows,(bbox[3]+1)*bl_cols]
    minr, minc, maxr, maxc = full_img_bbox
    maxr = min(maxr, img.shape[0]-1)
    maxc = min(maxc, img.shape[1]-1)

    return minr, minc, maxr, maxc


def check_breast_side_is_left_MG(image : np.ndarray, max_pixel_value = 255.0, dark_region_ratio = 15.0) -> bool:
    """
    checks if the breast is in the left of the image
    :param image: numpy image , expected 2d grayscale image
    :param max_pixel_value: maximum possible value in the image grayscale format
    :param dark_region_ratio: the raito of possible grayscale values which are considered dark
    :return: True iff the breast side is left
    """
    cols = image.shape[1]
    left_side = image[:, :cols // 2]
    right_side = image[:,cols//2:]
    dark_region = max_pixel_value / dark_region_ratio
    return np.count_nonzero(left_side<dark_region) < np.count_nonzero(right_side<dark_region)


def flip_if_needed(image : np.ndarray) -> np.ndarray :
    """
    checks if the image needed to be flipped to the left
    :param image: numpy image , expected 2d grayscale image
    :return: image where the breast is in the left
    """
    if not check_breast_side_is_left_MG(image): #orig
        image = np.fliplr(image)
    return image


def standardize_breast_image(inner_image : np.ndarray, normalized_target_range: Tuple[float, float]) -> np.ndarray :
    """
    standardize breast to be focused on the left and normalize the values from integer to float
    :param inner_image: numpy image , expected 2d breast integer grayscale image where 0 is black background color
    :param normalized_target_range: requested normalized image pixels range
    :return: numpy image 2d grayscale image , breast focused on the left
    """
    # flip if needed
    inner_image = flip_if_needed(inner_image)

    aabb = find_breast_aabb(inner_image)
    inner_image = inner_image[aabb[0]: aabb[2], aabb[1]: aabb[3]].copy()
    # normalize
    inner_image = FuseProcessorsImageToolBox.normalize_to_range(inner_image, range=normalized_target_range)

    return inner_image