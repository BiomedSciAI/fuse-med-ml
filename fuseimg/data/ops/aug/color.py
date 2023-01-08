from typing import List, Optional
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
from fuse.utils.rand.param_sampler import Gaussian
from fuse.data.ops.ops_cast import Cast
from fuseimg.data.ops.color import OpClip
from torch import Tensor
import torch


class OpAugColor(OpBase):
    """
    Color augmentation for gray scale images of any dimensions, including addition, multiplication, gamma and contrast adjusting
    """

    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this op expects torch tensor of range [0, 1]. Set to False to disable verification
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        add: Optional[float] = None,
        mul: Optional[float] = None,
        gamma: Optional[float] = None,
        contrast: Optional[float] = None,
        channels: Optional[List[int]] = None,
    ):
        """
        :param key: key to a image stored in sample_dict: torch tensor of range [0, 1] representing an image to ,
        :param add: value to add to each pixel
        :param mul: multiplication factor
        :param gamma: gamma factor
        :param contrast: contrast factor
        :param channels: Apply clipping just over the specified channels. If set to None will apply on all channels.
        """
        aug_input = sample_dict[key]

        # verify
        if self._verify_arguments:
            assert isinstance(aug_input, torch.Tensor), f"Error: OpAugColor expects torch Tensor, got {type(aug_input)}"
            assert (
                aug_input.min() >= 0.0 and aug_input.max() <= 1.0
            ), f"Error: OpAugColor expects tensor in range [0.0-1.0]. got [{aug_input.min()}-{aug_input.max()}]"

        aug_tensor = aug_input
        if channels is None:
            if add is not None:
                aug_tensor = self.aug_op_add_col(aug_tensor, add)
            if mul is not None:
                aug_tensor = self.aug_op_mul_col(aug_tensor, mul)
            if gamma is not None:
                aug_tensor = self.aug_op_gamma(aug_tensor, 1.0, gamma)
            if contrast is not None:
                aug_tensor = self.aug_op_contrast(aug_tensor, contrast)
        else:
            if add is not None:
                aug_tensor[channels] = self.aug_op_add_col(aug_tensor[channels], add)
            if mul is not None:
                aug_tensor[channels] = self.aug_op_mul_col(aug_tensor[channels], mul)
            if gamma is not None:
                aug_tensor[channels] = self.aug_op_gamma(aug_tensor[channels], 1.0, gamma)
            if contrast is not None:
                aug_tensor[channels] = self.aug_op_contrast(aug_tensor[channels], contrast)

        sample_dict[key] = aug_tensor
        return sample_dict

    @staticmethod
    def aug_op_add_col(aug_input: Tensor, add: float) -> Tensor:
        """
        Adding a values to all pixels
        :param aug_input: the tensor to augment
        :param add: the value to add to each pixel
        :return: the augmented tensor
        """
        aug_tensor = aug_input + add
        aug_tensor = OpClip.clip(aug_tensor, clip=(0.0, 1.0))
        return aug_tensor

    @staticmethod
    def aug_op_mul_col(aug_input: Tensor, mul: float) -> Tensor:
        """
        multiply each pixel
        :param aug_input: the tensor to augment
        :param mul: the multiplication factor
        :return: the augmented tensor
        """
        input_tensor = aug_input * mul
        input_tensor = OpClip.clip(input_tensor, clip=(0.0, 1.0))
        return input_tensor

    @staticmethod
    def aug_op_gamma(aug_input: Tensor, gain: float, gamma: float) -> Tensor:
        """
        Gamma augmentation
        :param aug_input: the tensor to augment
        :param gain: gain factor
        :param gamma: gamma factor
        :return: None
        """
        input_tensor = (aug_input**gamma) * gain
        input_tensor = OpClip.clip(input_tensor, clip=(0.0, 1.0))
        return input_tensor

    @staticmethod
    def aug_op_contrast(aug_input: Tensor, factor: float) -> Tensor:
        """
        Adjust contrast (notice - calculated across the entire input tensor, even if it's 3d)
        :param aug_input:the tensor to augment
        :param factor: contrast factor.   1.0 is neutral
        :return: the augmented tensor
        """
        calculated_mean = aug_input.mean()
        input_tensor = ((aug_input - calculated_mean) * factor) + calculated_mean
        input_tensor = OpClip.clip(input_tensor, clip=(0.0, 1.0))
        return input_tensor


class OpAugGaussian(OpBase):
    """
    Add gaussian noise to numpy array or torch tensor of any dimensions
    """

    def __call__(
        self, sample_dict: NDict, key: str, mean: float = 0.0, std: float = 0.03, channels: Optional[List[int]] = None
    ) -> Tensor:
        """
        :param key: key to a tensor or numpy array stored in sample_dict: any dimension and any range
        :param mean: mean gaussian distribution
        :param std:  std gaussian distribution
        :param channels: Apply just over the specified channels. If set to None will apply on all channels.
        """
        aug_input = sample_dict[key]

        aug_tensor = aug_input
        if channels is None:
            rand_patch = Gaussian(aug_tensor.shape, mean, std).sample()
            rand_patch = Cast.like(rand_patch, aug_tensor)
            aug_tensor = aug_tensor + rand_patch

        else:
            rand_patch = Gaussian(aug_tensor[channels].shape, mean, std).sample()
            rand_patch = Cast.like(rand_patch, aug_tensor)
            aug_tensor[channels] = aug_tensor[channels] + rand_patch

        sample_dict[key] = aug_tensor
        return sample_dict
