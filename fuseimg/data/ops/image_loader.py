import os
from fuse.data.ops.op_base import OpReversibleBase
from typing import Optional
import numpy as np
import nibabel as nib
from fuse.utils.ndict import NDict

from torchvision.io import read_image
from medpy.io import load
import pydicom


class OpLoadImage(OpReversibleBase):
    """
    Loads a medical image, currently supports:
            'nii', 'nib', 'jpg', 'jpeg', 'png', 'mha','dcm'
    """

    def __init__(self, dir_path: str, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path

    def __call__(
        self,
        sample_dict: NDict,
        op_id: Optional[str],
        key_in: str,
        key_out: str,
        key_metadata_out: Optional[str] = None,
        format: str = "infer",
    ):
        """
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out: the key name in sample_dict that holds the image
        :param key_metadata_out : the key to hold metadata dictionary
        """
        img_filename = os.path.join(self._dir_path, sample_dict[key_in])
        img_filename_suffix = img_filename.split(".")[-1]
        if (format == "infer" and img_filename_suffix in ["nii"]) or (format in ["nii", "nib"]):

            img = nib.load(img_filename)
            img_np = img.get_fdata()
            sample_dict[key_out] = img_np

        elif img_filename_suffix in ["jpg", "jpeg", "png"]:
            img = read_image(img_filename)
            img = img.float()
            img_np = img.numpy()
            sample_dict[key_out] = img_np

        elif (format == "infer" and img_filename_suffix in ["mha"]) or (format in ["mha"]):
            image_data, image_header = load(img_filename)
            sample_dict[key_out] = image_data
            if key_metadata_out is not None:
                sample_dict[key_metadata_out] = {
                    key: image_header.sitkimage.GetMetaData(key) for key in image_header.sitkimage.GetMetaDataKeys()
                }
        elif (format == "infer" and img_filename_suffix in ["dcm"]) or (format in ["dcm"]):
            dcm = pydicom.dcmread(img_filename)
            inner_image = dcm.pixel_array
            # convert to numpy
            img_np = np.asarray(inner_image)
            sample_dict[key_out] = img_np
            if key_metadata_out is not None:
                metadata = NDict()
                for key, field in key_metadata_out:
                    metadata[field] = dcm[key].value
                sample_dict[key_metadata_out] = metadata
        else:
            raise Exception(
                f"OpLoadImage: case format {format} and {img_filename_suffix} is not supported - filename {img_filename}"
            )

        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict
