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
from functools import partial
import os
from typing import Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
import skimage
import skimage.transform

from fuse.utils import NDict
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform
import wget

from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuse.data.ops.op_base import OpReversibleBase
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_common import OpLambda

from fuse.data.utils.sample import get_sample_id

from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpClip, OpToRange


class OpKits21SampleIDDecode(OpReversibleBase):
    """
    decodes sample id into image and segmentation filename
    """

    def __call__(self, sample_dict: NDict, op_id: Optional[str]) -> NDict:
        """ """
        sid = get_sample_id(sample_dict)

        img_filename_key = "data.input.img_path"
        sample_dict[img_filename_key] = os.path.join(sid, "imaging.nii.gz")

        seg_filename_key = "data.gt.seg_path"
        sample_dict[seg_filename_key] = os.path.join(sid, "aggregated_MAJ_seg.nii.gz")

        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict


def my_resize(input_tensor: torch.Tensor, resize_to: Tuple[int, int, int]) -> torch.Tensor:
    """
    Custom resize operation for the CT image
    """

    inner_image_height = input_tensor.shape[0]
    inner_image_width = input_tensor.shape[1]
    inner_image_depth = input_tensor.shape[2]
    h_ratio = resize_to[0] / inner_image_height
    w_ratio = resize_to[1] / inner_image_width
    if h_ratio >= 1 and w_ratio >= 1:
        resize_ratio_xy = min(h_ratio, w_ratio)
    elif h_ratio < 1 and w_ratio < 1:
        resize_ratio_xy = max(h_ratio, w_ratio)
    else:
        resize_ratio_xy = 1
    # resize_ratio_z = self.resize_to[2] / inner_image_depth
    if resize_ratio_xy != 1 or inner_image_depth != resize_to[2]:
        input_tensor = skimage.transform.resize(
            input_tensor,
            output_shape=(
                int(inner_image_height * resize_ratio_xy),
                int(inner_image_width * resize_ratio_xy),
                int(resize_to[2]),
            ),
            mode="reflect",
            anti_aliasing=True,
        )
    return input_tensor


class KITS21:
    """
    2021 Kidney and Kidney Tumor Segmentation Challenge Dataset
    KITS21 data pipeline implementation. See https://github.com/neheller/kits21
    Currently including only the image and segmentation map
    """

    # bump whenever the static pipeline modified
    KITS21_DATASET_VER = 0

    @staticmethod
    def download(path: str, cases: Optional[Union[int, List[int]]] = None) -> None:
        """
        :param cases: pass None (default) to download all 300 cases. OR
        pass a list of integers with cases num in the range [0,299]. OR
        pass a single int to download a single case
        """
        if cases is None:
            cases = list(range(300))
        elif isinstance(cases, int):
            cases = [cases]
        elif not isinstance(cases, list):
            raise Exception("Unsupported args! please provide None, int or list of ints")

        dl_dir = path

        for i in tqdm(cases, total=len(cases)):
            destination_dir = os.path.join(dl_dir, f"case_{i:05d}")
            os.makedirs(destination_dir, exist_ok=True)

            # imaging
            destination_file = os.path.join(destination_dir, "imaging.nii.gz")
            src = f"https://kits19.sfo2.digitaloceanspaces.com/master_{i:05d}.nii.gz"
            if not os.path.exists(destination_file):
                wget.download(src, destination_file)
            else:
                print(f"imaging.nii.gz number {i} was found")

            # segmentation
            seg_file = "aggregated_MAJ_seg.nii.gz"
            destination_file = os.path.join(destination_dir, seg_file)
            src = f"https://github.com/neheller/kits21/raw/master/kits21/data/case_{i:05d}/aggregated_MAJ_seg.nii.gz"
            if not os.path.exists(destination_file):
                wget.download(src, destination_file)
            else:
                print(f"{seg_file} number {i} was found")

    @staticmethod
    def sample_ids():
        """
        get all the sample ids in trainset
        sample_id is case_{id:05d} (for example case_00001 or case_00100)
        """
        return [f"case_{case_id:05d}" for case_id in range(300)]

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault(
            "static",
            [
                # decoding sample ID
                (
                    OpKits21SampleIDDecode(),
                    dict(),
                ),  # will save image and seg path to "data.input.img_path", "data.gt.seg_path"
                # loading data
                (OpLoadImage(data_path), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
                (OpLoadImage(data_path), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
                # fixed image normalization
                (OpClip(), dict(key="data.input.img", clip=(-500, 500))),
                (OpToRange(), dict(key="data.input.img", from_range=(-500, 500), to_range=(0, 1))),
                # transposing so the depth channel will be first
                (
                    OpLambda(partial(np.moveaxis, source=-1, destination=0)),
                    dict(key="data.input.img"),
                ),  # convert image from shape [H, W, D] to shape [D, H, W]
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline():
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        """
        repeat_for = [dict(key="data.input.img"), dict(key="data.gt.seg")]

        dynamic_pipeline = PipelineDefault(
            "dynamic",
            [
                # resize image to (110, 256, 256)
                (
                    OpRepeat(
                        OpLambda(func=partial(my_resize, resize_to=(110, 256, 256))), kwargs_per_step_to_add=repeat_for
                    ),
                    dict(),
                ),
                # Numpy to tensor
                (OpRepeat(OpToTensor(), kwargs_per_step_to_add=repeat_for), dict(dtype=torch.float32)),
                # affine transformation per slice but with the same arguments
                (
                    OpSampleAndRepeat(OpAugAffine2D(), kwargs_per_step_to_add=repeat_for),
                    dict(
                        rotate=Uniform(-180.0, 180.0),
                        scale=Uniform(0.8, 1.2),
                        flip=(RandBool(0.5), RandBool(0.5)),
                        translate=(RandInt(-15, 15), RandInt(-15, 15)),
                    ),
                ),
                # color augmentation - check if it is useful in CT images
                (
                    OpSample(OpAugColor()),
                    dict(
                        key="data.input.img",
                        gamma=Uniform(0.8, 1.2),
                        contrast=Uniform(0.9, 1.1),
                        add=Uniform(-0.01, 0.01),
                    ),
                ),
                # add channel dimension -> [C=1, D, H, W]
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
            ],
        )
        return dynamic_pipeline

    @staticmethod
    def dataset(
        data_path: str,
        cache_dir: str,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
    ) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param reset_cache: set to True tp reset the cache
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        """

        if sample_ids is None:
            sample_ids = KITS21.sample_ids()

        static_pipeline = KITS21.static_pipeline(data_path)
        dynamic_pipeline = KITS21.dynamic_pipeline()

        cacher = SamplesCacher(
            f"kits21_cache_ver{KITS21.KITS21_DATASET_VER}",
            static_pipeline,
            [cache_dir],
            restart_cache=reset_cache,
            workers=num_workers,
        )

        my_dataset = DatasetDefault(
            sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,
        )
        my_dataset.create()
        return my_dataset
