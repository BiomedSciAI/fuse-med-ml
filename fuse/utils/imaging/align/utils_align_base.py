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

from abc import ABC


class FuseAlignMapBase(ABC):
    def __init__(self):
        """
        AlignMap settings, e.g. number of iterations for an iterative algorithm.
        """
        pass

    def align(self, img1, img2):
        """
        Learn mapping between two images. This may be a computationally heavy step.
        Mapping is unidirectional, from coordinates in img1 to coordinates in img2.
        Examples:
            For feature-based methods, calculate local features
            For deep learning methods, forward pass images
            For key-points based methods, find image key points (e.g. nipple)
        :param img1: ndarray, float32
        :param img2: ndarray, float32
        :return: None
        """
        pass

    def translate_xy(self, x, y):
        """
        Map coordinates: img1 --> img2
        Ideally this should be a computationally light step.
        :param x: horizontal position, measured from left side of images
        :param y: vertical position, measured from top of image
        :return: translated coordinates with respect to img2
        """
        pass
