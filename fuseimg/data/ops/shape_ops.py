from typing import Optional
import numpy as np
from torch import Tensor


from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase

from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.data.ops.ops_common_imaging import OpApplyTypesImaging
import torch


def sanity_check_HWC(input_tensor):
    if 3 != input_tensor.ndim:
        raise Exception(f"expected 3 dim tensor, instead got {input_tensor.shape}")
    assert input_tensor.shape[2] < input_tensor.shape[0]
    assert input_tensor.shape[2] < input_tensor.shape[1]


def sanity_check_CHW(input_tensor):
    if 3 != input_tensor.ndim:
        raise Exception(f"expected 3 dim tensor, instead got {input_tensor.shape}")
    assert input_tensor.shape[0] < input_tensor.shape[1]
    assert input_tensor.shape[0] < input_tensor.shape[2]


class OpHWCToCHW(OpBase):
    """
    HWC (height, width, channel) to CHW (channel, height, width)
    """

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key: str) -> NDict:
        """
        :param key: key to torch tensor of shape [H, W, C]
        """
        input_tensor: Tensor = sample_dict[key]

        sanity_check_HWC(input_tensor)
        input_tensor = input_tensor.permute(dims=(2, 0, 1))
        sanity_check_CHW(input_tensor)

        sample_dict[key] = input_tensor
        return sample_dict


class OpCHWToHWC(OpBase):
    """
    CHW (channel, height, width) to HWC (height, width, channel)
    """

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key: str) -> NDict:
        """
        :param key: key to torch tensor of shape [C, H, W]
        """
        input_tensor: Tensor = sample_dict[key]

        sanity_check_CHW(input_tensor)
        input_tensor = input_tensor.permute(dims=(1, 2, 0))
        sanity_check_HWC(input_tensor)

        sample_dict[key] = input_tensor
        return sample_dict


class OpSelectSlice(OpBase):
    """
    select one slice from the input tensor,
    from the first dimmention of a >2 dimensional input
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self, sample_dict: NDict, op_id: Optional[str], key: str, slice_idx: int
    ):
        """
        :param slice_idx: the index of the selected slice from the 1st dimmention of an input tensor
        """

        img = sample_dict[key]
        if len(img.shape) < 3:
            return sample_dict

        img = img[slice_idx]
        sample_dict[key] = img
        return sample_dict


op_select_slice_img_and_seg = OpApplyTypesImaging(
    {
        DataTypeImaging.IMAGE: (OpSelectSlice(), {}),
        DataTypeImaging.SEG: (OpSelectSlice(), {}),
    }
)
