
from typing import Optional, List
import numpy as np
from torch import Tensor
import torchvision.transforms.functional as TTF

from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase

from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.data.ops.ops_common_imaging import OpApplyTypesImaging
import torch

def sanity_check_HWC(input_tensor):
    if 3!=input_tensor.ndim:
        raise Exception(f'expected 3 dim tensor, instead got {input_tensor.shape}')
    assert input_tensor.shape[2]<input_tensor.shape[0]
    assert input_tensor.shape[2]<input_tensor.shape[1]

def sanity_check_CHW(input_tensor):
    if 3!=input_tensor.ndim:
        raise Exception(f'expected 3 dim tensor, instead got {input_tensor.shape}')
    assert input_tensor.shape[0]<input_tensor.shape[1]
    assert input_tensor.shape[0]<input_tensor.shape[2]


class OpHWCToCHW(OpBase):
    """
    HWC (height, width, channel) to CHW (channel, height, width)
    """
    
    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        '''
        :param key: key to torch tensor of shape [H, W, C]
        '''
        input_tensor: Tensor = sample_dict[key]

        sanity_check_HWC(input_tensor)
        input_tensor = input_tensor.permute(dims = (2, 0, 1))
        sanity_check_CHW(input_tensor)
        
        sample_dict[key] = input_tensor
        return sample_dict


class OpCHWToHWC(OpBase):
    """
    CHW (channel, height, width) to HWC (height, width, channel)
    """
    
    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        '''
        :param key: key to torch tensor of shape [C, H, W]
        '''
        input_tensor: Tensor = sample_dict[key]

        sanity_check_CHW(input_tensor)
        input_tensor = input_tensor.permute(dims = (1, 2, 0))
        sanity_check_HWC(input_tensor)
        
        sample_dict[key] = input_tensor
        return sample_dict


class OpSelectSlice(OpBase):
    '''
     select one slice from the input tensor, 
     from the first dimmention of a >2 dimensional input
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key: str,
        slice_idx: int
        ):
        '''
        :param slice_idx: the index of the selected slice from the 1st dimmention of an input tensor
        ''' 
        
        img = sample_dict[key]
        if len(img.shape) < 3:
            return sample_dict

        img = img[slice_idx]
        sample_dict[key] = img
        return sample_dict

op_select_slice_img_and_seg = OpApplyTypesImaging({DataTypeImaging.IMAGE : (OpSelectSlice(), {}),
                                DataTypeImaging.SEG : (OpSelectSlice(), {}) })
        

class OpPad(OpBase):
    """
    Pad the give image on all the sides. Supports Tensor & ndarray.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key: str,
            padding: List[int],
            fill: int = 0,
            mode: str = 'constant',
            **kwargs):
        """
        Pad values
        :param key: key to an image in sample_dict - either torch tensor or ndarray
        :param padding: padding on each border. can be differerniate each border by passing a list.
        :param fill: if mode = 'constant', pads with fill's value.
        :param padding_mode: see torch's & numpy's pad functions for more details.
        :param kwargs: numpy's pad function give supports to more arguments. See it's docs for more details.
        """

        img = sample_dict[key]
        
        if torch.is_tensor(img):
            processed_img = TTF.pad(img, padding, fill, mode)

        elif isinstance(img, np.ndarray):
            # kwargs['constant_values'] = fill
            processed_img = np.pad(img, pad_width=padding, mode=mode, constant_values=fill, **kwargs)

        else:
            raise Exception(f"Error: OpPad expects Tensor or nd.array object, but got {type(img)}.")

        sample_dict[key] = processed_img
        return sample_dict
