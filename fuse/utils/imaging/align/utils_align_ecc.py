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

from fuse.utils.align.utils_align_base import FuseAlignMapBase
import numpy as np
import cv2


class FuseAlignMapECC(FuseAlignMapBase):
    def __init__(self, transformation_type='homography', num_iterations=600, termination_eps=1e-4):
        transformation_type = transformation_type.lower()
        assert transformation_type in ['homography', 'affine']

        self.transformation_type = transformation_type.lower()
        self.num_iterations = num_iterations
        self.termination_eps = termination_eps
        self.transformation_matrix = None
        self.img1_aligned = None
        self.img2_aligned = None

    def align(self, img1, img2, mask=None):

        if self.transformation_type == 'affine':
            transformation = cv2.MOTION_AFFINE
            M = np.eye(2, 3, dtype=np.float32)
        elif self.transformation_type == 'homography':
            transformation = cv2.MOTION_HOMOGRAPHY
            M = np.eye(3, 3, dtype=np.float32)
        else:
            raise Exception(
                f"Unknown transformation {self.transformation_type}. Allowed values are : ['affine', 'homography']")

        # convert from [0, 1] to binary uint8
        img1_original = img1.copy()
        img2_original = img2.copy()
        img1 = ((img1 > 0) * 255).astype(np.uint8)
        img2 = ((img2 > 0) * 255).astype(np.uint8)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.num_iterations, self.termination_eps)

        s, M = cv2.findTransformECC(img1, img2, M, transformation, criteria=criteria, inputMask=mask)

        self.transformation_matrix = M

        # Apply the alignment on img1 and the inverse alignment on img2
        if transformation == cv2.MOTION_AFFINE:
            self.img1_aligned = cv2.warpAffine(img1_original, self.transformation_matrix, (img2.shape[1], img2.shape[0]),
                                                    flags=cv2.INTER_LINEAR)
            self.img2_aligned = cv2.warpAffine(img2_original, self.transformation_matrix, (img1.shape[1], img1.shape[0]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            self.img1_aligned = cv2.warpPerspective(img1_original, self.transformation_matrix,
                                                    (img2.shape[1], img2.shape[0]),
                                                    flags=cv2.INTER_LINEAR)
            self.img2_aligned = cv2.warpPerspective(img2_original, self.transformation_matrix,
                                                    (img1.shape[1], img1.shape[0]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        self.mask = mask

    def translate_xy(self, x, y):
        if self.transformation_matrix is None:
            raise Exception("No transformation matrix found - need to run 'align' first!")

        if self.transformation_type == 'homography':
            wx = (self.transformation_matrix[0, 0] * x + self.transformation_matrix[0, 1] * y + self.transformation_matrix[
                0, 2]) / (self.transformation_matrix[2, 0] * x + self.transformation_matrix[2, 1] * y +
                          self.transformation_matrix[2, 2])
            wy = (self.transformation_matrix[1, 0] * x + self.transformation_matrix[1, 1] * y + self.transformation_matrix[
                1, 2]) / (self.transformation_matrix[2, 0] * x + self.transformation_matrix[2, 1] * y +
                          self.transformation_matrix[2, 2])
        else:
            raise Exception('Currently supports only homography translation')

        return wx, wy
