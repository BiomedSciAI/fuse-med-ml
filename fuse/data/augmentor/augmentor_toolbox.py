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

from copy import deepcopy
from typing import Tuple, Any, List, Iterable, Optional

import numpy
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch import Tensor

from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerGaussianPatch as Gaussian
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform


######## Affine augmentation
def aug_op_affine(aug_input: Tensor, rotate: float = 0.0, translate: Tuple[float, float] = (0.0, 0.0),
                  scale: Tuple[float, float] = 1.0, flip: Tuple[bool, bool] = (False, False), shear: float = 0.0,
                  channels: Optional[List[int]] = None) -> Tensor:
    """
    Affine augmentation
    :param aug_input: 2D tensor representing an image to augment, shape [num_channels, height, width] or [height, width]
    :param rotate: angle [0.0 - 360.0]
    :param translate: translation per spatial axis (number of pixels). The sign used as the direction.
    :param scale: scale factor
    :param flip: flip per spatial axis flip[0] for vertical flip and flip[1] for horizontal flip
    :param shear: shear factor
    :param channels: apply the augmentation on the specified channels. Set to None to apply to all channels.
    :return: the augmented image
    """
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

    return aug_tensor


def aug_op_affine_group(aug_input: Tuple[Tensor], **kwargs) -> Tuple[Tensor]:
    """
    Applies same augmentation on multiple tensors. For example, augmenting both input image and its corresponding
    segmentation mask in the same way. This method wraps 'aug_op_affine'.
    :param aug_input: tuple of tensors
    :param kwargs:    augmentation params, same kwargs as 'aug_op_affine' - see docstring there
    :return: tuple of tensors, all augmented the same way
    """
    return tuple((aug_op_affine(element, **kwargs) for element in aug_input))


def aug_op_crop_and_resize(aug_input: Tensor,
                           scale: Tuple[float, float],
                           channels: Optional[List[int]] = None) -> Tensor:
    """
    Alternative to rescaling: center crop and resize back to the original dimensions. if scale is bigger than 1.0. the image first padded.
    :param aug_input: The tensor to augment
    :param scale: tuple of positive floats
    :param channels: apply augmentation on the specified channels or None for all of them
    :return: the augmented tensor
    """
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

    return aug_tensor


######## Color augmentation
def aug_op_clip(aug_input: Tensor, clip: Tuple[float, float] = (-1.0, 1.0)) -> Tensor:
    """
    Clip pixel values
    :param aug_input: the tensor to clip
    :param clip: values for clipping from both sides
    :return: Clipped tensor
    """
    aug_tensor = aug_input
    if clip is not None:
        aug_tensor = torch.clamp(aug_tensor, clip[0], clip[1], out=aug_tensor)
    return aug_tensor


def aug_op_add_col(aug_input: Tensor, add: float) -> Tensor:
    """
    Adding a values to all pixels
    :param aug_input: the tensor to augment
    :param add: the value to add to each pixel
    :return: the augmented tensor
    """
    aug_tensor = aug_input + add
    aug_tensor = aug_op_clip(aug_tensor, clip=(0, 1))
    return aug_tensor


def aug_op_mul_col(aug_input: Tensor, mul: float) -> Tensor:
    """
    multiply each pixel
    :param aug_input: the tensor to augment
    :param mul: the multiplication factor
    :return: the augmented tensor
    """
    input_tensor = aug_input * mul
    input_tensor = aug_op_clip(input_tensor, clip=(0, 1))
    return input_tensor


def aug_op_gamma(aug_input: Tensor, gain: float, gamma: float) -> Tensor:
    """
    Gamma augmentation
    :param aug_input: the tensor to augment
    :param gain: gain factor
    :param gamma: gamma factor
    :return: None
    """
    input_tensor = (aug_input ** gamma) * gain
    input_tensor = aug_op_clip(input_tensor, clip=(0, 1))
    return input_tensor


def aug_op_contrast(aug_input: Tensor, factor: float) -> Tensor:
    """
    Adjust contrast (notice - calculated across the entire input tensor, even if it's 3d)
    :param aug_input:the tensor to augment
    :param factor: contrast factor.   1.0 is neutral
    :return: the augmented tensor
    """
    calculated_mean = aug_input.mean()
    input_tensor = ((aug_input - calculated_mean) * factor) + calculated_mean
    input_tensor = aug_op_clip(input_tensor, clip=(0, 1))
    return input_tensor


def aug_op_color(aug_input: Tensor, add: Optional[float] = None, mul: Optional[float] = None,
                 gamma: Optional[float] = None, contrast: Optional[float] = None, channels: Optional[List[int]] = None):
    """
    Color augmentaion: including addition, multiplication, gamma and contrast adjusting
    :param aug_input: the tensor to augment
    :param add: value to add to each pixel
    :param mul: multiplication factor
    :param gamma: gamma factor
    :param contrast: contrast factor
    :param channels: Apply clipping just over the specified channels. If set to None will apply on all channels.
    :return:
    """
    aug_tensor = aug_input
    if channels is None:
        if add is not None:
            aug_tensor = aug_op_add_col(aug_tensor, add)
        if mul is not None:
            aug_tensor = aug_op_mul_col(aug_tensor, mul)
        if gamma is not None:
            aug_tensor = aug_op_gamma(aug_tensor, 1.0, gamma)
        if contrast is not None:
            aug_tensor = aug_op_contrast(aug_tensor, contrast)
    else:
        if add is not None:
            aug_tensor[channels] = aug_op_add_col(aug_tensor[channels], add)
        if mul is not None:
            aug_tensor[channels] = aug_op_mul_col(aug_tensor[channels], mul)
        if gamma is not None:
            aug_tensor[channels] = aug_op_gamma(aug_tensor[channels], 1.0, gamma)
        if contrast is not None:
            aug_tensor[channels] = aug_op_contrast(aug_tensor[channels], contrast)

    return aug_tensor


######## Gaussian noise
def aug_op_gaussian(aug_input: Tensor, mean: float = 0.0, std: float = 0.03, channels: Optional[List[int]] = None) -> Tensor:
    """
    Add gaussian noise
    :param aug_input: the tensor to augment
    :param mean: mean gaussian distribution
    :param std:  std gaussian distribution
    :param channels: Apply just over the specified channels. If set to None will apply on all channels.
    :return: the augmented tensor
    """
    aug_tensor = aug_input
    if channels is None:
        rand_patch = Gaussian(aug_tensor.shape, mean, std).sample()
        aug_tensor = aug_tensor + rand_patch
    else:
        rand_patch = Gaussian(aug_tensor[channels].shape, mean, std).sample()
        aug_tensor[channels] = aug_tensor[channels] + rand_patch
    return aug_tensor


def aug_op_elastic_transform(aug_input: Tensor, alpha: float = 1, sigma: float = 50, channels: Optional[List[int]] = None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis",
       :param aug_input: input tensor of shape (C,Y,X)
       :param alpha: global pixel shifting (correlated to the article)
       :param sigma: Gaussian filter parameter
       :param channels: which channels to apply the augmentation
       :return distorted image
    """
    random_state = numpy.random.RandomState(None)
    if channels is None:
        channels = list(range(aug_input.shape[0]))
    aug_tensor = aug_input.numpy()
    for channel in channels:
        aug_channel_tensor = aug_input[channel].numpy()
        shape = aug_channel_tensor.shape
        dx1 = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dx2 = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x1, x2 = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        indices = numpy.reshape(x2 + dx2, (-1, 1)), numpy.reshape(x1 + dx1, (-1, 1))

        distored_image = map_coordinates(aug_channel_tensor, indices, order=1, mode='reflect')
        distored_image = distored_image.reshape(aug_channel_tensor.shape)
        aug_tensor[channel] = distored_image
    return torch.from_numpy(aug_tensor)


######### Default / Example augmentation pipline for a 2D image
def aug_image_default_pipeline(input_pointer: str) -> List[Any]:
    """
    Return default image augmentation pipeline. optimised for breast project (GMP model).
    In case paramter tunning is required - copy and change the values
    :param input_pointer: global dict pointer to the image
    :return: the default pipeline
    """
    return [
        [
            (input_pointer,),
            aug_op_affine,
            {'rotate': Uniform(-30.0, 30.0), 'translate': (RandInt(-10, 10), RandInt(-10, 10)),
             'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
            {'apply': RandBool(0.5)}
        ],
        [
            (input_pointer,),
            aug_op_color,
            {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
             'contrast': Uniform(0.85, 1.15)},
            {'apply': RandBool(0.5)}
        ],
        [
            (input_pointer,),
            aug_op_gaussian,
            {'std': 0.03},
            {'apply': RandBool(0.5)}
        ],
    ]


# general utilities
def aug_pipeline_step_replicate(step: List, key: str, values: Iterable) -> List[List]:
    """
    Replicate a step, but set different value for each replication for the specified key
    :param step: The step to replicate
    :param key: the key to override (withing te augmentation dunction input)
    :param values: Iterable specify the value for each replication
    :return:
    """
    list_of_steps = []
    for value in values:
        step_copy = deepcopy(step)
        step_copy[2][key] = value
        list_of_steps.append(step_copy)

    return list_of_steps


def aug_op_rescale_pixel_values(aug_input: Tensor, target_range: Tuple[float, float] = (-1.0, 1.0)) -> Tensor:
    """
    Scales pixel values to specific range.
    :param aug_input:       input tensor
    :param target_range:    target range, (min, max)
    :return:  rescaled tensor
    """
    max_val = aug_input.max()
    min_val = aug_input.min()
    if min_val == max_val == 0:
        return aug_input
    aug_input = aug_input - min_val
    aug_input = aug_input / (max_val - min_val)
    aug_input = aug_input * (target_range[1] - target_range[0])
    aug_input = aug_input + target_range[0]
    return aug_input


def squeeze_3d_to_2d(aug_input: Tensor, axis_squeeze: str) -> Tensor:
    '''
    squeeze selected axis of volume image into channel dimension, in
    order to fit the 2D augmentation functions
    :param aug_input: input of shape: (channel, z, y, x)
    :return:
    '''
    # aug_input shape is [channels, z, y, x]
    if axis_squeeze == 'y':
        aug_input = aug_input.permute((0, 2, 1, 3))
        # aug_input shape is [channels, y, z, x]
    elif axis_squeeze == 'x':
        aug_input = aug_input.permute((0, 3, 2, 1))
        # aug_input shape is [channels, x, y, z]
    else:
        assert axis_squeeze == 'z', "axis squeeze must be a string of either x, y, or z"
    return aug_input.reshape((aug_input.shape[0] * aug_input.shape[1],) + aug_input.shape[2:])


def unsqueeze_2d_to_3d(aug_input: Tensor, channels: int, axis_squeeze: str) -> Tensor:
    '''
    unsqueeze selected axis to original shape, and add the batch dimension
    :param aug_input:
    :return:
    '''
    aug_input = aug_input
    aug_input = aug_input.reshape((channels, aug_input.shape[0] // channels) + aug_input.shape[1:])
    if axis_squeeze == 'y':
        aug_input = aug_input.permute((0, 2, 1, 3))
        # aug_input shape is [channels, z, y, x]
    elif axis_squeeze == 'x':
        aug_input = aug_input.permute((0, 3, 2, 1))
        # aug_input shape is [channels, z, y, x]
    else:
        assert axis_squeeze == 'z', "axis squeeze must be a string of either x, y, or z"
    return aug_input


def rotation_in_3d(aug_input: Tensor, z_rot: float = 0.0, y_rot: float = 0.0, x_rot: float = 0):
    """
    rotates an input tensor around an axis, when for example z_rot is chosen,
    the rotation is in the x-y plane.
    Note: rotation angles are in relation to the original axis (not the rotated one)
    rotation angles should be given in degrees
    :param aug_input:image input should be in shape [channel, z, y, x]
    :param z_rot: angle to rotate x-y plane clockwise
    :param y_rot: angle to rotate x-z plane clockwise
    :param x_rot: angle to rotate z-y plane clockwise
    :return:
    """
    assert len(aug_input.shape) == 4  # will only work for 3d
    channels = aug_input.shape[0]
    if z_rot != 0:
        squeez_img = squeeze_3d_to_2d(aug_input, axis_squeeze='z')
        rot_squeeze = aug_op_affine(squeez_img, rotate=z_rot)
        aug_input = unsqueeze_2d_to_3d(rot_squeeze, channels, 'z')
    if x_rot != 0:
        squeez_img = squeeze_3d_to_2d(aug_input, axis_squeeze='x')
        rot_squeeze = aug_op_affine(squeez_img, rotate=x_rot)
        aug_input = unsqueeze_2d_to_3d(rot_squeeze, channels, 'x')
    if y_rot != 0:
        squeez_img = squeeze_3d_to_2d(aug_input, axis_squeeze='y')
        rot_squeeze = aug_op_affine(squeez_img, rotate=y_rot)
        aug_input = unsqueeze_2d_to_3d(rot_squeeze, channels, 'y')

    return aug_input


def aug_cut_out(aug_input: Tensor, fill: float = None, size: int = 16) -> Tensor:
    """
    removing small patch of the image. https://arxiv.org/abs/1708.04552
    :param aug_input: the tensor to augment
    :param fill: value to fill the patch
    :param size:  size of patch
    :return: the augmented tensor
    """
    fill = aug_input.mean(-1).mean(-1) if fill is None else fill
    sx = torch.randint(0, aug_input.shape[1] - size, (1,))
    sy = torch.randint(0, aug_input.shape[2] - size, (1,))
    aug_input[:, sx:sx + size, sy:sy + size] = fill[:, None, None]

    return aug_input


def aug_op_batch_mix_up(aug_input: Tuple[Tensor, Tensor], factor: float) -> Tuple[Tensor, Tensor]:
    """
    mixup augmentation on a batch level
    :param aug_input: batch level input to augment. tuple of image and one hot vector of targets
    :param factor: background factor
    :return: the augmented batch
    """
    img = aug_input[0]
    labels = aug_input[1]
    perm = numpy.arange(img.shape[0])
    numpy.random.shuffle(perm)
    img_mix_up = img[perm]
    labels_mix_up = labels[perm]
    img = img * (1.0 - factor) + factor * img_mix_up
    labels = labels * (1.0 - factor) + factor * labels_mix_up
    return img, labels
