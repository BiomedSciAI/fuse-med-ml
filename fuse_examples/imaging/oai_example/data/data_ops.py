from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
import numpy as np
import os
from copy import deepcopy
import random
import pydicom
from scipy.ndimage import zoom
import torch.nn.functional as F
from typing import Tuple, Union, List
from volumentations import Compose
from functools import reduce


class OpLoadData(OpBase):
    """
    loads 3D image from a folder of dicom files.
    Folder should have only dicom files and they should be names in the order of the sequence.
    """

    def __init__(self, im2D: bool = False, path_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.im2D = im2D
        self.path_key = path_key

    def __call__(self, sample_dict: NDict) -> NDict:
        """ """
        folder_path = sample_dict[self.path_key]
        dicom_files = [f for f in os.listdir(folder_path)]

        # Sort the DICOM files based on their file names (assuming they are numbered in order)
        dicom_files.sort()
        # Load the first DICOM file to get the image dimensions
        file_path = os.path.join(folder_path, dicom_files[0])
        dicom = pydicom.dcmread(file_path)
        rows, cols = dicom.pixel_array.shape
        num_slices = len(dicom_files)

        if self.im2D:
            random_idx = random.randint(0, num_slices - 1)
            dicom = pydicom.dcmread(os.path.join(folder_path, dicom_files[random_idx]))
            sample_dict["img"] = dicom.pixel_array
            return sample_dict

        # Create an empty 3D NumPy array to store the DICOM series
        dicom_array = np.zeros((num_slices, rows, cols), dtype=dicom.pixel_array.dtype)

        # Load each DICOM file and store it in the 3D array
        for i, file_name in enumerate(dicom_files):
            file_path = os.path.join(folder_path, file_name)
            dicom = pydicom.dcmread(file_path)
            dicom_array[i] = dicom.pixel_array

        sample_dict["img"] = dicom_array

        return sample_dict


class OpNormalizeMRI(OpBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        to_range: Tuple[float, float],
        max_key: str = None,
    ):

        img = sample_dict[key]
        img = np.clip(
            img, *(np.percentile(img, [0, 95]))
        )  # truncate the intensities to the range of 0.5 to 99.5 percentiles
        if max_key is None:
            from_range_start = img.min()
            from_range_end = img.max()
        else:
            from_range_start = 0
            from_range_end = sample_dict[max_key]
        to_range_start = to_range[0]
        to_range_end = to_range[1]

        # shift to start at 0
        img -= from_range_start
        if (from_range_end - from_range_start) == 0:
            print("MRI bad range")
            return None
        # scale to be in desired range
        img *= (to_range_end - to_range_start) / (from_range_end - from_range_start)
        # shift to start in desired start val
        img += to_range_start

        sample_dict[key] = img

        return sample_dict


class OpResize3D(OpBase):
    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        shape: Tuple[int, int, int],
        segmentation: bool = False,
    ) -> Union[None, dict, List[dict]]:
        depth, height, width = sample_dict[key].shape
        depth_factor = shape[0] / depth
        height_factor = shape[1] / height
        width_factor = shape[2] / width
        if segmentation:
            sample_dict[key] = zoom(
                sample_dict[key],
                (depth_factor, height_factor, width_factor),
                order=0,
                mode="nearest",
            )
        else:
            sample_dict[key] = zoom(
                sample_dict[key], (depth_factor, height_factor, width_factor), order=1
            )
        return sample_dict


class OpPickSlice(OpBase):
    """ """

    def __init__(
        self,
        img_key: str = "img",
        seg_key: str = "seg",
        slice_key: str = "slice",
        drop_blank: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_key = img_key
        self.seg_key = seg_key
        self.slice_key = slice_key
        self.drop_blank = drop_blank

    def __call__(self, sample_dict: NDict) -> NDict:

        sample_dict[self.img_key] = sample_dict[self.img_key][
            sample_dict[self.slice_key]
        ]
        sample_dict[self.seg_key] = sample_dict[self.seg_key][
            sample_dict[self.slice_key]
        ]
        if self.drop_blank:
            if sample_dict[self.seg_key][1:].max() < 1:
                return None

        return sample_dict


class OpDinoCrops(OpBase):
    """
    This function takes in a dictionary containing an image and a key, and returns a new dictionary with multiple crops of the original image.

    Parameters:
        sample_dict (NDict): A dictionary containing an image and a key.
        key (str): The key of the image in the dictionary.
        n_crops (int): The number of crops to create.

    Returns:
        NDict: A new dictionary with multiple crops of the original image.
    """

    def __call__(
        self, sample_dict: NDict, key: str, n_crops: int
    ) -> Union[None, dict, List[dict]]:
        """ """
        img = sample_dict[key]
        for i in range(n_crops):
            sample_dict[f"crop_{i}"] = deepcopy(img)

        del sample_dict[key]

        # sample_dict["data.input.img"] = np.moveaxis((np.repeat(img[...,np.newaxis],3,-1)), -1, 0)
        return sample_dict


class OpRandomCrop(OpBase):
    """
    Resizes the input image or volume to the specified size.

    Args:
        sample_dict (NDict): A dictionary containing the input image or volume.
        key (str): The key of the input image or volume in the sample_dict.
        scale (tuple, optional): A tuple specifying the minimum and maximum scaling factors. Defaults to (0.4, 1.0).
        on_depth (bool, optional): A boolean flag indicating whether to crop also the depth dimension. Defaults to True.
        res_shape (list, optional): A list specifying the target resolution. If provided, overrides the scale parameter. Defaults to None.

    Returns:
        NDict: A dictionary containing the resized image or volume.
    """

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        scale: Tuple[float, float] = (0.4, 1.0),
        on_depth: bool = True,
        res_shape: List = None,
    ) -> Union[None, dict, List[dict]]:
        """ """
        if isinstance(key, str):
            key = [key]
        shape = sample_dict[key[0]].shape
        if scale is not None:
            scale = random.uniform(scale[0], scale[1])
            scaled_shape = [round(scale * d) for d in shape]
        elif res_shape is not None:
            scaled_shape = res_shape
        starts = [random.randint(0, f - s) for (f, s) in zip(shape, scaled_shape)]
        for k in key:
            img = sample_dict[k]
            assert len(img.shape) == 3
            if on_depth:
                sample_dict[k] = img[
                    starts[0] : starts[0] + scaled_shape[0],
                    starts[1] : starts[1] + scaled_shape[1],
                    starts[2] : starts[2] + scaled_shape[2],
                ]
            else:
                sample_dict[k] = img[
                    :,
                    starts[1] : starts[1] + scaled_shape[1],
                    starts[2] : starts[2] + scaled_shape[2],
                ]
        return sample_dict


class OpRandomFlip(OpBase):
    def __call__(self, sample_dict: NDict, key: str) -> Union[None, dict, List[dict]]:
        """ """
        if isinstance(key, str):
            key = [key]
        n_axes = len(sample_dict[key[0]].shape)
        bool_rand_vec = np.random.choice([True, False], size=n_axes)
        for k in key:
            sample_dict[k] = np.flip(
                sample_dict[k], axis=np.where(bool_rand_vec)[0]
            ).copy()

        return sample_dict


class OpVolumentation(OpBase):
    def __init__(self, compose: Compose):
        super().__init__()
        self.compose = compose

    def __call__(self, sample_dict: NDict, key: str) -> Union[None, dict, List[dict]]:
        img = {"image": sample_dict[key]}
        sample_dict[key] = self.compose(**img)["image"]
        return sample_dict


class OpMask3D(OpBase):
    def __init__(
        self,
        mask_percentage: float = 0.3,
        cuboid_size: Union[List[int], tuple] = (2, 2, 2),
    ):
        super().__init__()
        assert 0 <= mask_percentage <= 1, "Mask percentage must be between 0 and 1"
        self.mask_percentage = mask_percentage
        self.cuboid_size = cuboid_size

    def generate_block_positions(self, shape: tuple) -> np.ndarray:
        """Generate a grid of possible block positions"""
        if len(shape) == 3:
            z_positions = range(
                0, shape[0] - self.cuboid_size[0] + 1, self.cuboid_size[0]
            )
            y_positions = range(
                0, shape[1] - self.cuboid_size[1] + 1, self.cuboid_size[1]
            )
            x_positions = range(
                0, shape[2] - self.cuboid_size[2] + 1, self.cuboid_size[2]
            )

            positions = np.array(
                [
                    (z, y, x)
                    for z in z_positions
                    for y in y_positions
                    for x in x_positions
                ]
            )
        else:
            y_positions = range(
                0, shape[0] - self.cuboid_size[0] + 1, self.cuboid_size[0]
            )
            x_positions = range(
                0, shape[1] - self.cuboid_size[1] + 1, self.cuboid_size[1]
            )

            positions = np.array([(y, x) for y in y_positions for x in x_positions])

        return positions

    def apply_block_mask(self, mask: np.ndarray, position: tuple) -> None:
        """Apply mask to a single block"""
        if len(mask.shape) == 3:
            z, y, x = position
            mask[
                z : z + self.cuboid_size[0],
                y : y + self.cuboid_size[1],
                x : x + self.cuboid_size[2],
            ] = True
        else:
            y, x = position
            mask[y : y + self.cuboid_size[0], x : x + self.cuboid_size[1]] = True

    def __call__(self, sample_dict: NDict, key: str, out_key: str) -> NDict:
        img = sample_dict[key]
        assert len(img.shape) in [2, 3], "Input must be 2D or 3D array"

        # Create empty mask
        mask = np.zeros_like(img, dtype=bool)

        # Get all possible block positions
        positions = self.generate_block_positions(img.shape)

        # Calculate number of blocks needed
        total_pixels = reduce(lambda x, y: x * y, img.shape)
        block_volume = reduce(lambda x, y: x * y, self.cuboid_size)
        num_blocks = len(positions)
        target_blocks = int(np.ceil(self.mask_percentage * total_pixels / block_volume))

        # Randomly select blocks to mask
        selected_indices = np.random.choice(
            num_blocks, size=min(target_blocks, num_blocks), replace=False
        )

        # Apply masks for selected blocks
        for idx in selected_indices:
            self.apply_block_mask(mask, positions[idx])

        # Apply the mask
        masked_image = img.copy()
        masked_image[mask] = 0

        # Calculate actual masked percentage
        # actual_percentage = np.mean(mask)

        # Store results
        sample_dict[out_key] = masked_image
        # sample_dict[f"{out_key}_mask_percentage"] = actual_percentage

        return sample_dict


class OpSegToOneHot(OpBase):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, sample_dict: NDict, key) -> NDict:
        seg_tensor = sample_dict[key]
        seg_tensor = seg_tensor.squeeze(0).long()
        one_hot = F.one_hot(seg_tensor, num_classes=self.n_classes)
        one_hot = one_hot.permute(-1, *list(range(len(one_hot.shape) - 1)))

        sample_dict[key] = one_hot
        return sample_dict
