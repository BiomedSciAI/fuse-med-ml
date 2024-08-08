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


class OpLoadData(OpBase):
    """
    loads 3D image from a folder of dicom files.
    Folder should have only dicom files and they should be names in the order of the sequence.
    """

    def __init__(self, path_key: str = None):
        super().__init__()
        self.path_key = path_key

    def __call__(self, sample_dict: NDict) -> Union[None, dict, List[dict]]:
        """ """
        folder_path = sample_dict[self.path_key]
        dicom_files = [f for f in os.listdir(folder_path)]

        # Sort the DICOM files based on their file names (assuming they are numbered in order)
        dicom_files.sort()
        # Load the first DICOM file to get the image dimensions
        file_path = os.path.join(folder_path, dicom_files[0])
        dicom = pydicom.dcmread(file_path)
        rows, cols = dicom.pixel_array.shape

        # Create an empty 3D NumPy array to store the DICOM series
        num_slices = len(dicom_files)
        dicom_array = np.zeros((num_slices, rows, cols), dtype=dicom.pixel_array.dtype)

        # Load each DICOM file and store it in the 3D array
        for i, file_name in enumerate(dicom_files):
            file_path = os.path.join(folder_path, file_name)
            dicom = pydicom.dcmread(file_path)
            dicom_array[i] = dicom.pixel_array

        sample_dict["img"] = dicom_array

        return sample_dict


class OpNormalizeMRI(OpBase):
    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        to_range: Tuple[float, float],
    ) -> Union[None, dict, List[dict]]:

        img = sample_dict[key]
        img = np.clip(
            img, *(np.percentile(img, [0, 95]))
        )  # truncate the intensities to the range of 0.5 to 99.5 percentiles

        from_range_start = img.min()
        from_range_end = img.max()
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

        flip_depth = np.random.choice([True, False])
        flip_spatial = np.random.choice([True, False])
        for k in key:
            if flip_depth:
                sample_dict[k] = np.flip(sample_dict[k], axis=0).copy()

            if flip_spatial:
                # Flip both height and width to maintain square shape
                sample_dict[k] = np.flip(sample_dict[k], axis=(1, 2)).copy()

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
        cuboid_size: Tuple[int, int, int] = [2, 2, 2],
    ) -> Union[None, dict, List[dict]]:
        super().__init__()
        self.mask_percentage = mask_percentage
        self.cuboid_size = cuboid_size

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        img = sample_dict[key]

        depth, height, width = img.shape

        # Calculate the number of cuboids needed to mask the specified percentage
        total_voxels = depth * height * width
        cuboid_volume = self.cuboid_size[0] * self.cuboid_size[1] * self.cuboid_size[2]
        num_cuboids = int((self.mask_percentage / 100) * total_voxels / cuboid_volume)

        masked_image = img.copy()

        for _ in range(num_cuboids):
            # Generate random starting coordinates for the cuboid
            z = np.random.randint(0, depth - self.cuboid_size[0] + 1)
            y = np.random.randint(0, height - self.cuboid_size[1] + 1)
            x = np.random.randint(0, width - self.cuboid_size[2] + 1)

            # Mask the cuboid region
            masked_image[
                z : z + self.cuboid_size[0],
                y : y + self.cuboid_size[1],
                x : x + self.cuboid_size[2],
            ] = 0

        sample_dict[key] = masked_image

        return sample_dict


class OpSegToOneHot(OpBase):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, sample_dict: NDict, key: str) -> Union[None, dict, List[dict]]:
        seg_tensor = sample_dict[key]
        seg_tensor = seg_tensor.squeeze(0).long()
        one_hot = F.one_hot(seg_tensor, num_classes=self.n_classes)
        one_hot = one_hot.permute(3, 0, 1, 2)
        sample_dict[key] = one_hot
        return sample_dict
