
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

from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
from fuse.data.utils.sample import get_sample_id

# from fuse.data.processor.processor_base import FuseProcessorBase


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


class OpImageMaskLoader(OpBase):
    def __init__(self,
                 data_csv: str = None,
                 size: int = 512,
                 normalization: float = 255.0, **kwargs):
        """
        Create Input processor
        :param input_data:              path to images
        :param normalized_target_range: range for image normalization
        :param resize_to:               Optional, new size of input images, keeping proportions
        """
        super().__init__(**kwargs)

        if data_csv:
            self.df = pd.read_csv(data_csv)
        else:
            self.df = None

        self.size = (size, size)
        self.norm = normalization

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key_in:str, key_out: str):

        desc = get_sample_id(sample_dict)

        if self.df is not None:  # compute mask 
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

        else:  # load image
            dcm = pydicom.read_file(desc).pixel_array
            image = np.asarray(PIL.Image.fromarray(dcm).resize(self.size))

            image = image.astype('float32')
            image = image / 255.0

        # convert image from shape (H x W x C) to shape (C x H x W) with C=3
        if len(image.shape) > 2:
            image = np.moveaxis(image, -1, 0)
        else:
            image = np.expand_dims(image, 0)

        # numpy to tensor
        # sample = torch.from_numpy(image)

        # except:
        #     return None

        sample_dict[key_out] = image
        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:        
        return sample_dict          
