from typing import Tuple, Union
import numpy as np
import torch
from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase

from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.data.ops.ops_common_imaging import OpApplyTypesImaging


class OpClip(OpBase):
    """
    Clip values - support both torch tensor and numpy array
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        clip=(0.0, 1.0),
    ):
        """
        Clip  values
        :param key: key to an image in sample_dict: either torh tensor or numpy array and any dimension
        :param clip: values for clipping from both sides
        """

        img = sample_dict[key]

        processed_img = self.clip(img, clip)

        sample_dict[key] = processed_img
        return sample_dict

    @staticmethod
    def clip(
        img: Union[np.ndarray, torch.Tensor], clip: Tuple[float, float] = (0.0, 1.0)
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(img, np.ndarray):
            processed_img = np.clip(img, clip[0], clip[1])
        elif isinstance(img, torch.Tensor):
            processed_img = torch.clamp(img, clip[0], clip[1], out=img)
        else:
            raise Exception(f"Error: unexpected type {type(img)}")
        return processed_img


op_clip_img = OpApplyTypesImaging({DataTypeImaging.IMAGE: (OpClip(), {})})


class OpNormalizeAgainstSelf(OpBase):
    """
    normalizes a tensor into [0.0, 1.0] using its own statistics (NOT against a dataset)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key: str):
        img = sample_dict[key].astype(np.float32)
        img -= img.min()
        img /= img.max()
        sample_dict[key] = img

        return sample_dict


op_normalize_against_self_img = OpApplyTypesImaging({DataTypeImaging.IMAGE: (OpNormalizeAgainstSelf(), {})})


class OpToIntImageSpace(OpBase):
    """
    normalizes a tensor into [0, 255] int gray-scale using its own statistics (NOT against a dataset)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
    ):
        img = sample_dict[key]
        img -= img.min()
        img /= img.max()
        img *= 255.0
        img = img.astype(np.uint8).copy()
        # img = img.transpose((1, 2, 0))
        sample_dict[key] = img
        return sample_dict


op_to_int_image_space_img = OpApplyTypesImaging({DataTypeImaging.IMAGE: (OpToIntImageSpace(), {})})


class OpToRange(OpBase):
    """
    linearly project from a range to a different range
    """

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        from_range: Tuple[float, float],
        to_range: Tuple[float, float],
    ):

        from_range_start = from_range[0]
        from_range_end = from_range[1]
        to_range_start = to_range[0]
        to_range_end = to_range[1]

        img = sample_dict[key]

        # shift to start at 0
        img -= from_range_start

        # scale to be in desired range
        img *= (to_range_end - to_range_start) / (from_range_end - from_range_start)
        # shift to start in desired start val
        img += to_range_start

        sample_dict[key] = img

        return sample_dict


op_to_range_img = OpApplyTypesImaging({DataTypeImaging.IMAGE: (OpToRange(), {})})
