from typing import List, Optional, Tuple, Union

from PIL import Image

import numpy as np
import torch
import torchvision.transforms.functional as TTF
import torchvision.transforms as transforms
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

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        rotate: float = 0.0,
        translate: Tuple[float, float] = (0.0, 0.0),
        scale: Tuple[float, float] = 1.0,
        flip: Tuple[bool, bool] = (False, False),
        shear: float = 0.0,
        channels: Optional[List[int]] = None,
        interpolation: int = transforms.InterpolationMode.BILINEAR,
    ) -> Union[None, dict, List[dict]]:
        """
        :param key: key to a tensor stored in sample_dict: 2D tensor representing an image to augment, shape [num_channels, height, width] or [height, width]
        :param rotate: angle [-360.0 - 360.0]
        :param translate: translation per spatial axis (number of pixels). The sign used as the direction.
        :param scale: scale factor
        :param flip: flip per spatial axis flip[0] for vertical flip and flip[1] for horizontal flip
        :param shear: shear factor
        :param channels: apply the augmentation on the specified channels. Set to None to apply to all channels.
        :param interpolation: used PIL interpolation method see TTF.affine for details.
        :return: the augmented image
        """
        aug_input = sample_dict[key]

        # verify
        if self._verify_arguments:
            assert isinstance(
                aug_input, torch.Tensor
            ), f"Error: OpAugAffine2D expects torch Tensor, got {type(aug_input)}"
            assert len(aug_input.shape) in [
                2,
                3,
            ], f"Error: OpAugAffine2D expects tensor with 2 or 3 dimensions. got {aug_input.shape}"

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
            aug_channel_tensor = TTF.affine(
                aug_channel_tensor,
                angle=rotate,
                scale=scale,
                interpolation=interpolation,
                translate=translate,
                shear=shear,
            )
            if flip[0]:
                aug_channel_tensor = TTF.vflip(aug_channel_tensor)
            if flip[1]:
                aug_channel_tensor = TTF.hflip(aug_channel_tensor)

            # convert back to torch tensor
            aug_channel_tensor = np.array(aug_channel_tensor)
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

    def __call__(
        self, sample_dict: NDict, key: str, scale: Tuple[float, float], channels: Optional[List[int]] = None
    ) -> Union[None, dict, List[dict]]:
        """
        :param key: key to a tensor stored in sample_dict: 2D tensor representing an image to augment, shape [num_channels, height, width] or [height, width]
        :param scale: tuple of positive floats
        :param channels: apply augmentation on the specified channels or None for all of them
        :return: the augmented tensor
        """
        aug_input = sample_dict[key]

        # verify
        if self._verify_arguments:
            assert isinstance(
                aug_input, torch.Tensor
            ), f"Error: OpAugCropAndResize2D expects torch Tensor, got {type(aug_input)}"
            assert len(aug_input.shape) in [
                2,
                3,
            ], f"Error: OpAugCropAndResize2D expects tensor with 2 or 3 dimensions. got {aug_input.shape}"

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
                cropped_shape = (
                    int(aug_channel_tensor.shape[0] * scale[0]),
                    int(aug_channel_tensor.shape[1] * scale[1]),
                )
                padding = [[0, 0], [0, 0]]
                for dim in range(2):
                    if scale[dim] > 1.0:
                        padding[dim][0] = (cropped_shape[dim] - aug_channel_tensor.shape[dim]) // 2
                        padding[dim][1] = (cropped_shape[dim] - aug_channel_tensor.shape[dim]) - padding[dim][0]
                aug_channel_tensor_pad = TTF.pad(
                    aug_channel_tensor.unsqueeze(0), (padding[1][0], padding[0][0], padding[1][1], padding[0][1])
                )
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
            assert isinstance(
                aug_input, torch.Tensor
            ), f"Error: OpAugSqueeze3Dto2D expects torch Tensor, got {type(aug_input)}"
            assert (
                len(aug_input.shape) == 4
            ), f"Error: OpAugSqueeze3Dto2D expects tensor with 4 dimensions. got {aug_input.shape}"

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

        aug_output = aug_input.reshape((aug_input.shape[0] * aug_input.shape[1],) + aug_input.shape[2:])

        sample_dict[key] = aug_output
        return sample_dict


class OpAugUnsqueeze3DFrom2D(OpBase):
    """
    Unsqueeze selected axis of volume image from channel dimension, restore the original shape squeezed by OpAugSqueeze3Dto2D
    """

    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this ops expects torch tensor with 2 dimensions. Set to False to disable verification
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(self, sample_dict: NDict, key: str, axis_squeeze: int, channels: int) -> NDict:
        """
        :param key: key to a tensor stored in sample_dict and squeezed by OpAugSqueeze3Dto2D
        :param axis_squeeze: axis squeeze as specified in OpAugSqueeze3Dto2D
        :param channels: number of channels in the original tensor (before OpAugSqueeze3Dto2D)
        """
        aug_input = sample_dict[key]

        # verify
        if self._verify_arguments:
            assert isinstance(
                aug_input, torch.Tensor
            ), f"Error: OpAugUnsqueeze3DFrom2D expects torch Tensor, got {type(aug_input)}"
            assert (
                len(aug_input.shape) == 3
            ), f"Error: OpAugUnsqueeze3DFrom2D expects tensor with 3 dimensions. got {aug_input.shape}"

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


class OpCrop3D(OpBase):
    """
    crop to certain size. if the image is smaller than the size then its padded.
    """

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        output_shape: Tuple[int, int, int],
        z_move=0.5,
        x_move=0.5,
        y_move=0.5,
        fill: int = 0,
    ):
        """
        :param key: key to a tensor stored in sample_dict and get cropped by OpRandomCrop3D
        :param out_size: shape of the output tensor
        :param fill: if the image needs padding then it will be filled with that value
        :param z_move: float between 0 and 1 which is the proportion of the movement that can be done (for x and y is the same idea).
        when they are all set to 0.5 it means we are cropping from the center of the image.
        """
        aug_input = sample_dict[key]
        assert len(aug_input.shape) == len(output_shape)
        assert z_move >= 0 and z_move <= 1
        assert x_move >= 0 and x_move <= 1
        assert y_move >= 0 and y_move <= 1
        depth, height, width = aug_input.shape  # input is in the form [D,H,W]

        aug_tensor = torch.full(output_shape, fill, dtype=torch.float32)

        if depth > output_shape[0]:
            crop_start = round(z_move * (depth - output_shape[0]))
            aug_input = aug_input[crop_start : crop_start + output_shape[0], :, :]
        if height > output_shape[1]:
            crop_start = round(y_move * (height - output_shape[1]))
            aug_input = aug_input[:, crop_start : crop_start + output_shape[1], :]
        if width > output_shape[2]:
            crop_start = round(x_move * (width - output_shape[2]))
            aug_input = aug_input[:, :, crop_start : crop_start + output_shape[2]]

        aug_tensor[:depth, :height, :width] = aug_input
        sample_dict[key] = aug_tensor

        return sample_dict


class OpResizeTo(OpBase):
    """
    Resizes an image into the given dimensions. Currently supports only ndarray
    """

    def __init__(self, channels_first: bool):
        """
        :param channels_first: assign True iff the input is in CxHxW format.
        """
        super().__init__()
        self._channels_first = channels_first

    def __call__(self, sample_dict: NDict, output_shape: Tuple[int], key: str, **kwargs) -> NDict:
        """
        :param key: key to a numpy array or tensor stored in the sample_dict in a H x W x C format.
        :param kwargs: additional arguments to pass to the resize function

        Stores the resized image in sample_dict[key]
        """
        aug_input = sample_dict[key]
        dim = len(aug_input.shape)

        if self._channels_first:
            # Permutes CxHxW -> HxWxC (for skimage's resize)
            perm = self.get_permutation(dim=dim, channels_first=True)
            aug_input = np.transpose(aug_input, axes=perm)

        # Apply Resize
        aug_output = skimage.transform.resize(image=aug_input, output_shape=output_shape, **kwargs)

        if self._channels_first:
            # Permutes back HxWxC -> CxHxW
            perm = self.get_permutation(dim=dim, channels_first=False)
            aug_output = np.transpose(aug_output, axes=perm)

        sample_dict[key] = aug_output

        return sample_dict

    def get_permutation(self, dim: int, channels_first: bool):
        """
        :param dim: tensor's dimension
        :param channels_first: True iff the wanted permutation is: HxWxC -> CxHxW

        Returns the right permutation to:
        channels_first is True -> converting from CxHxW to HxWxC
            i.e: (dim-2, dim-1, 0, 1, ..., dim-3)
        channels_first is False -> converting from HxWxC to CxHxW
            i.e: (2, 3, ..., dim-2, dim-1, 0, 1)
        """

        if dim < 2:
            raise Exception(f"Error, dim ({dim}) must be greater or equal to 2.")

        if channels_first:
            channels = [0]
            hw = [i for i in range(1, dim)]
            perm = hw + channels

        else:
            channels = [dim - 1]
            hw = [i for i in range(0, dim - 1)]
            perm = channels + hw

        perm = tuple(perm)
        return perm


class OpRotation3D(OpBase):
    def __call__(self, sample_dict: NDict, key: str, z_rot: float = 0.0, y_rot: float = 0.0, x_rot: float = 0):
        """
        rotates an input tensor around an axis, when for example z_rot is chosen,
        the rotation is in the x-y plane.
        Note: rotation angles are in relation to the original axis (not the rotated one)
        rotation angles should be given in degrees
        :param aug_input:image input should be in shape [z, y, x] [depth, height, width]
        :param z_rot: angle to rotate x-y plane clockwise
        :param y_rot: angle to rotate x-z plane clockwise
        :param x_rot: angle to rotate z-y plane clockwise
        :return:
        """
        aug_input = sample_dict[key]
        assert len(aug_input.shape) == 3  # will only work for 3d
        if z_rot != 0:
            aug_input = TTF.rotate(aug_input, angle=z_rot)
        if x_rot != 0:
            aug_input = aug_input.permute(dims=(2, 0, 1))
            aug_input = TTF.rotate(aug_input, angle=x_rot)
            aug_input = aug_input.permute(dims=(1, 2, 0))
        if y_rot != 0:
            aug_input = aug_input.permute(dims=(1, 0, 2))
            aug_input = TTF.rotate(aug_input, angle=y_rot)
            aug_input = aug_input.permute(dims=(1, 0, 2))

        sample_dict[key] = aug_input
        return sample_dict
