from typing import Any, Callable, Tuple, Union
import numpy as np
import torch
from fuse.utils.ndict import NDict
import SimpleITK as sitk

from fuse.data.ops.op_base import OpBase

from fuseimg.utils.typing.key_types_imaging import DataTypeImaging
from fuseimg.data.ops.ops_common_imaging import OpApplyTypesImaging


class OpClip(OpBase):
    """
    Clip values - supports: torch Tensor, numpy array and SITK Image
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
        elif isinstance(img, sitk.Image):
            ref = img
            img = sitk.GetArrayFromImage(img)
            processed_img = np.clip(img, clip[0], clip[1])
            processed_img = sitk.GetImageFromArray(processed_img)
            processed_img.CopyInformation(ref)
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
        img = sample_dict[key]

        is_stk_image = isinstance(img, sitk.Image)
        if is_stk_image:
            ref = img
            img = sitk.GetArrayFromImage(img)
            img = img.astype(float)

        img -= img.min()
        img /= img.max()

        if is_stk_image:
            img = sitk.GetImageFromArray(img)
            img.CopyInformation(ref)

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

        img = sample_dict[key]
        is_stk_image = isinstance(img, sitk.Image)

        if is_stk_image:
            ref = img
            img = sitk.GetArrayFromImage(img)
            img = img.astype(float)

        img = self.to_range(img, from_range, to_range)

        if is_stk_image:
            img = sitk.GetImageFromArray(img)
            img.CopyInformation(ref)

        sample_dict[key] = img
        return sample_dict

    @staticmethod
    def to_range(img: np.ndarray, from_range: Tuple[float, float], to_range: Tuple[float, float]):
        from_range_start = from_range[0]
        from_range_end = from_range[1]
        to_range_start = to_range[0]
        to_range_end = to_range[1]

        # shift to start at 0
        img -= from_range_start

        # scale to be in desired range
        img *= (to_range_end - to_range_start) / (from_range_end - from_range_start)
        # shift to start in desired start val
        img += to_range_start

        return img


def call_on_stk_img(img: sitk.Image, func: Callable, **func_kwargs: Any):
    """
    Apply function on SITK Image object while keeping the image information

    :param img: the SITK Image object
    :param func: the function we want to apply on the Image
    :param func_kwargs: function's arguments
    """
    # Save ref Image to keep the info
    ref = img

    # Convert to ndarray and apply the function
    img = sitk.GetArrayFromImage(img)
    img = img.astype(float)
    img = func(img, **func_kwargs)

    # Return to SITK Image and copy info from ref
    img = sitk.GetImageFromArray(img)
    img.CopyInformation(ref)

    return img


op_to_range_img = OpApplyTypesImaging({DataTypeImaging.IMAGE: (OpToRange(), {})})
