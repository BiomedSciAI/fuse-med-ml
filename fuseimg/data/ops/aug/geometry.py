from typing import List, Optional, Tuple, Union

from torch import Tensor
from PIL import Image

import numpy
import torch
import torchvision.transforms.functional as TTF
import skimage
import skimage.transform

from fuse.utils.ndict import NDict

from fuse.data import OpBase

class OpAugAffine2D(OpBase):
    """
    2D affine transformation 
    """
    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this op expects torch tensor with either 2 or 3 dimensions. Set to False to disable verification 
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(self, sample_dict: NDict, key: str, rotate: float = 0.0, translate: Tuple[float, float] = (0.0, 0.0),
                    scale: Tuple[float, float] = 1.0, flip: Tuple[bool, bool] = (False, False), shear: float = 0.0,
                    channels: Optional[List[int]] = None) -> Union[None, dict, List[dict]]:
        """
        :param key: key to a tensor stored in sample_dict: 2D tensor representing an image to augment, shape [num_channels, height, width] or [height, width]
        :param rotate: angle [-360.0 - 360.0]
        :param translate: translation per spatial axis (number of pixels). The sign used as the direction.
        :param scale: scale factor
        :param flip: flip per spatial axis flip[0] for vertical flip and flip[1] for horizontal flip
        :param shear: shear factor
        :param channels: apply the augmentation on the specified channels. Set to None to apply to all channels.
        :return: the augmented image
        """
        aug_input = sample_dict[key]
        
        # verify
        if self._verify_arguments:
            assert isinstance(aug_input, torch.Tensor), f"Error: OpAugAffine2D expects torch Tensor, got {type(aug_input)}"
            assert len(aug_input.shape) in [2, 3], f"Error: OpAugAffine2D expects tensor with 2 or 3 dimensions. got {aug_input.shape}"

        # Support for 2D inputs - implicit single channel
        if len(aug_input.shape) == 2:
            aug_input = aug_input.unsqueeze(dim=0)
            remember_to_squeeze = True
        else:
            remember_to_squeeze = False

        # convert to PIL (required by affine augmentation function)
        if channels is None:
            channels = list(range(aug_input.shape[0]))
        aug_tensor = aug_input
        for channel in channels:
            aug_channel_tensor = aug_input[channel].numpy()
            aug_channel_tensor = Image.fromarray(aug_channel_tensor)
            aug_channel_tensor = TTF.affine(aug_channel_tensor, angle=rotate, scale=scale, translate=translate, shear=shear)
            if flip[0]:
                aug_channel_tensor = TTF.vflip(aug_channel_tensor)
            if flip[1]:
                aug_channel_tensor = TTF.hflip(aug_channel_tensor)

            # convert back to torch tensor
            aug_channel_tensor = numpy.array(aug_channel_tensor)
            aug_channel_tensor = torch.from_numpy(aug_channel_tensor)

            # set the augmented channel
            aug_tensor[channel] = aug_channel_tensor

        # squeeze back to 2-dim if needed
        if remember_to_squeeze:
            aug_tensor = aug_tensor.squeeze(dim=0)

        sample_dict[key] = aug_tensor
        return sample_dict


class OpAugCropAndResize2D(OpBase):
    """
    Alternative to rescaling in OpAugAffine2D: center crop and resize back to the original dimensions. if scale is bigger than 1.0. the image first padded.
    """
    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this ops expects torch tensor with either 2 or 3 dimensions. Set to False to disable verification 
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(self, sample_dict: NDict, key: str,
                                scale: Tuple[float, float],
                                channels: Optional[List[int]] = None) ->  Union[None, dict, List[dict]]:
        """
        :param key: key to a tensor stored in sample_dict: 2D tensor representing an image to augment, shape [num_channels, height, width] or [height, width]
        :param scale: tuple of positive floats
        :param channels: apply augmentation on the specified channels or None for all of them
        :return: the augmented tensor
        """
        aug_input = sample_dict[key]
        
        # verify
        if self._verify_arguments:
            assert isinstance(aug_input, torch.Tensor), f"Error: OpAugCropAndResize2D expects torch Tensor, got {type(aug_input)}"
            assert len(aug_input.shape) in [2, 3], f"Error: OpAugCropAndResize2D expects tensor with 2 or 3 dimensions. got {aug_input.shape}"
        
        if len(aug_input.shape) == 2:
            aug_input = aug_input.unsqueeze(dim=0)
            remember_to_squeeze = True
        else:
            remember_to_squeeze = False

        if channels is None:
            channels = list(range(aug_input.shape[0]))
        aug_tensor = aug_input
        for channel in channels:
            aug_channel_tensor = aug_input[channel]

            if scale[0] != 1.0 or scale[1] != 1.0:
                cropped_shape = (int(aug_channel_tensor.shape[0] * scale[0]), int(aug_channel_tensor.shape[1] * scale[1]))
                padding = [[0, 0], [0, 0]]
                for dim in range(2):
                    if scale[dim] > 1.0:
                        padding[dim][0] = (cropped_shape[dim] - aug_channel_tensor.shape[dim]) // 2
                        padding[dim][1] = (cropped_shape[dim] - aug_channel_tensor.shape[dim]) - padding[dim][0]
                aug_channel_tensor_pad = TTF.pad(aug_channel_tensor.unsqueeze(0), (padding[1][0], padding[0][0], padding[1][1], padding[0][1]))
                aug_channel_tensor_cropped = TTF.center_crop(aug_channel_tensor_pad, cropped_shape)
                aug_channel_tensor = TTF.resize(aug_channel_tensor_cropped, aug_channel_tensor.shape).squeeze(0)
                # set the augmented channel
                aug_tensor[channel] = aug_channel_tensor

        # squeeze back to 2-dim if needed
        if remember_to_squeeze:
            aug_tensor = aug_tensor.squeeze(dim=0)

        sample_dict[key] = aug_tensor
        return sample_dict


class OpAugSqueeze3Dto2D(OpBase):
    """
    Squeeze selected axis of volume image into channel dimension, in order to fit the 2D augmentation functions
    """
    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this ops expects torch tensor with 4 dimensions. Set to False to disable verification 
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(self, sample_dict: NDict, key: str, axis_squeeze: int) -> NDict:
        """
        :param key: key to a tensor stored in sample_dict: 3D tensor representing an image to augment, shape [num_channels, spatial axis 1, spatial axis 2, spatial axis 3]
        :param axis_squeeze: the axis (1, 2 or 3) to squeeze into channel dimension - typically z axis
        """
        aug_input = sample_dict[key]
        
        # verify
        if self._verify_arguments:
            assert isinstance(aug_input, torch.Tensor), f"Error: OpAugSqueeze3Dto2D expects torch Tensor, got {type(aug_input)}"
            assert len(aug_input.shape) == 4, f"Error: OpAugSqueeze3Dto2D expects tensor with 4 dimensions. got {aug_input.shape}"
        
        # aug_input shape is [channels, axis_1, axis_2, axis_3]
        if axis_squeeze == 1:
            pass
        elif axis_squeeze == 2:
            aug_input = aug_input.permute((0, 2, 1, 3))
            # aug_input shape is [channels, axis_2, axis_1, axis_3]
        elif axis_squeeze == 3:
            aug_input = aug_input.permute((0, 3, 1, 2))
            # aug_input shape is [channels, axis_3, axis_1, axis_2]
        else:
            raise Exception(f"Error: axis squeeze must be 1, 2, or 3, got {axis_squeeze}")
        
        aug_output =  aug_input.reshape((aug_input.shape[0] * aug_input.shape[1],) + aug_input.shape[2:])
        
        sample_dict[key] = aug_output
        return sample_dict

class OpAugUnsqueeze3DFrom2D(OpBase):
    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this ops expects torch tensor with 2 dimensions. Set to False to disable verification 
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    """
    Unsqueeze selected axis of volume image from channel dimension, restore the original shape squeezed by OpAugSqueeze3Dto2D
    """
    def __call__(self, sample_dict: NDict, key: str, axis_squeeze: int, channels: int) -> NDict:
        """
        :param key: key to a tensor stored in sample_dict and squeezed by OpAugSqueeze3Dto2D
        :param axis_squeeze: axis squeeze as specified in OpAugSqueeze3Dto2D
        :param channels: number of channels in the original tensor (before OpAugSqueeze3Dto2D)
        """
        aug_input = sample_dict[key]

        # verify
        if self._verify_arguments:
            assert isinstance(aug_input, torch.Tensor), f"Error: OpAugUnsqueeze3DFrom2D expects torch Tensor, got {type(aug_input)}"
            assert len(aug_input.shape) == 3, f"Error: OpAugUnsqueeze3DFrom2D expects tensor with 3 dimensions. got {aug_input.shape}"

        aug_output = aug_input.reshape((channels, aug_input.shape[0] // channels) + aug_input.shape[1:])
        
        if axis_squeeze == 1:
            pass
        elif axis_squeeze == 2:
            # aug_output shape is [channels, axis_2, axis_1, axis_3]
            aug_output = aug_output.permute((0, 2, 1, 3))
            # aug_input shape is [channels, axis 1, axis 2, axis 3]
        elif axis_squeeze == 3:
            # aug_output shape is [channels, axis_3, axis_1, axis_2]
            aug_output = aug_output.permute((0, 2, 3, 1))
            # aug_input shape is [channels, axis 1, axis 2, axis 3]
        else:
            raise Exception(f"Error: axis squeeze must be 1, 2, or 3, got {axis_squeeze}")
        
        sample_dict[key] = aug_output
        return sample_dict


class OpResizeTo(OpBase):
    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: 
        
        This Op expects torch / ndarray with d dimensions such that len(resize_to) == d.
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key: str, resize_to: List[int], mode: str='reflect', anti_aliasing: bool=True) -> NDict:
        """
        :param key: key to a numpy array or tensor stored in the sample_dict
        :param resize_to: the desired dimensions
        :param mode:
        :param anti_aliasing:

        Outputs a numpy array to 'sample_dict[key]'
        """
        aug_input = sample_dict[key]

        # verify arguments
        if self._verify_arguments:
            input_dim = len(aug_input.shape)
            resize_dim = len(resize_to)
            assert isinstance(aug_input, numpy.ndarray) or isinstance(aug_input, torch.Tensor), f"Error: OpResizeTo expects torch Tensor, got {type(aug_input)}"
            assert input_dim == resize_dim, f"Error, OpResizeTo expects the user to specify values for all the dimensions. got tensor with {input_dim} but {resize_dim} dimensions were given"
        
        aug_output = skimage.transform.resize(image=aug_input, output_shape=resize_to, mode=mode, anti_aliasing=anti_aliasing)
        sample_dict[key] = aug_output

        return sample_dict
