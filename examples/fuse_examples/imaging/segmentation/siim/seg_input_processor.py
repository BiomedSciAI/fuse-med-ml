
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
import pandas as pd
from skimage.io import imread
import torch
from pathlib import Path
import PIL
import pydicom

from typing import Optional, Tuple

from fuse.data.processor.processor_base import FuseProcessorBase


def rle2mask(rles, width, height):
    """
    
    rle encoding if images
    input: rles(list of rle), width and height of image
    returns: mask of shape (width,height)
    """
    
    mask= np.zeros(width* height)
    for rle in rles:
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]

    return mask.reshape(width, height).T


class SegInputProcessor(FuseProcessorBase):
    def __init__(self,
                 name: str = 'image', # can be 'image' or 'mask'
                 data_csv: str = None,
                 size: int = 512,
                 normalization: float = 255.0,
                 ):
        """
        Create Input processor
        :param input_data:              path to images
        :param normalized_target_range: range for image normalization
        :param resize_to:               Optional, new size of input images, keeping proportions
        :param padding:                 Optional, padding size
        """
        self.name = name
        assert self.name == 'image' or self.name == 'mask', "Error: name can be image or mask only."

        if data_csv:
            self.df = pd.read_csv(data_csv)

        self.size = (size, size)
        self.norm = normalization

    def __call__(self,
                 desc,
                 *args, **kwargs):

        try:

            if self.name == 'image':
                dcm = pydicom.read_file(desc).pixel_array
                image = np.asarray(PIL.Image.fromarray(dcm).resize(self.size))

                image = image.astype('float32')
                image = image / 255.0

            else: # create mask
                I = self.df.ImageId == Path(desc).stem
                enc = self.df.loc[I, ' EncodedPixels']
                if sum(I) == 0:
                    im = np.zeros((1024, 1024)).astype(np.uint8)
                elif sum(I) == 1:
                    enc = enc.values[0]
                    if enc == '-1': 
                        im = np.zeros((1024, 1024)).astype(np.uint8)
                    else:
                        im = rle2mask([enc], 1024, 1024).astype(np.uint8)
                else:
                    im = rle2mask(enc.values, 1024, 1024).astype(np.uint8)

                im = np.asarray(PIL.Image.fromarray(im).resize(self.size))
                image = im > 0
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
