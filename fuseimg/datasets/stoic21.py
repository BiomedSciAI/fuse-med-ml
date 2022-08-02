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
from glob import glob
import os
from typing import Hashable, Optional, Sequence, Tuple


import numpy as np
import torch
import skimage
import skimage.transform


from fuse.utils import NDict
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform

from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_common import OpConcat, OpLambda, OpLookup, OpToOneHot
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToFloat, OpToInt, OpToNumpy
from fuse.data.utils.sample import get_sample_id

from fuseimg.data.ops.aug.geometry import OpAugAffine2D
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpClip, OpToRange


class OpSTOIC21SampleIDDecode(OpBase):
    """
    decodes sample id into image and segmentation filename
    """

    def __call__(self, sample_dict: NDict) -> NDict:
        """ """
        sid = get_sample_id(sample_dict)

        img_filename_key = "data.input.img_path"
        sample_dict[img_filename_key] = sid

        return sample_dict


class STOIC21:
    """
    Dataset created for COVID-19 AI challenge - https://stoic2021.grand-challenge.org/
    Aims to predict the severe outcome of COVID-19, based on the largest dataset of Computed Tomography (CT) images of COVID-19
    Each sample also include age, gender and targets rt-pcr result and outcome at 1 month: severe or non=severe. More details and download instructions can be found here: https://stoic2021.grand-challenge.org/stoic-db/.
    """

    # bump whenever the static pipeline modified
    STOIC21_DATASET_VER = 0

    @staticmethod
    def download(path: str) -> None:
        """
        Automatic download is not supported, please follow instructions in STOIC21 class header to download
        """
        assert (
            len(STOIC21.sample_ids(path)) > 0
        ), "automatic download is not supported, please follow instructions in STOIC21 class header to download"

    @staticmethod
    def sample_ids(path: str):
        """
        get all the sample ids in train-set
        sample_id is *.mha file found in the specified path
        """
        files = [os.path.join("data/mha/", os.path.basename(f)) for f in glob(os.path.join(path, "data/mha/*.mha"))]
        assert len(files) > 0, f"Expecting mha files in {os.path.join(path, 'data/mha/*.mha')}"
        return files

    @staticmethod
    def static_pipeline(data_path: str, output_shape: Tuple[int, int, int]) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original STOIC21 data (See in STOIC21 header the instructions to download)
        :param output_shape: fixed shape to resize the image to
        """
        static_pipeline = PipelineDefault(
            "stoic21_static",
            [
                # decoding sample ID
                (
                    OpSTOIC21SampleIDDecode(),
                    dict(),
                ),  # will save image and seg path to "data.input.img_path", "data.gt.seg_path"
                # loading data
                (
                    OpLoadImage(data_path),
                    dict(key_in="data.input.img_path", key_out="data.input.img", key_metadata_out="data.metadata"),
                ),
                # resize
                # transposing so the depth channel will be first
                (
                    OpLambda(partial(np.moveaxis, source=-1, destination=0)),
                    dict(key="data.input.img"),
                ),  # convert image from shape [H, W, D] to shape [D, H, W]
                (
                    OpLambda(
                        partial(
                            skimage.transform.resize,
                            output_shape=output_shape,
                            mode="reflect",
                            anti_aliasing=True,
                            preserve_range=True,
                        )
                    ),
                    dict(key="data.input.img"),
                ),
                # read labels
                (OpToInt(), dict(key="data.metadata.PatientID")),
                (
                    OpReadDataframe(
                        data_filename=os.path.join(data_path, "metadata/reference.csv"),
                        key_column="data.gt.PatientID",
                        key_name="data.metadata.PatientID",
                        rename_columns={
                            "PatientID": "data.gt.PatientID",
                            "probCOVID": "data.gt.probCOVID",
                            "probSevere": "data.gt.probSevere",
                        },
                    ),
                    dict(),
                ),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(train: bool, clip_range: Tuple[float, float]):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param clip_range: clip the original voxels values to fit this range
        """
        age_map = {"035Y": 0, "045Y": 1, "055Y": 2, "065Y": 3, "075Y": 4, "085Y": 5}
        gender_map = {"F": 0, "M": 1}
        dynamic_pipeline = PipelineDefault(
            "stoic21_dynamic",
            [
                # cast thickness to float
                (OpToFloat(), dict(key="data.metadata.SliceThickness")),
                # map input to categories
                (OpLookup(age_map), dict(key_in="data.metadata.PatientAge", key_out="data.input.age")),
                (OpToOneHot(len(age_map)), dict(key_in="data.input.age", key_out="data.input.age_one_hot")),
                (OpLookup(gender_map), dict(key_in="data.metadata.PatientSex", key_out="data.input.gender")),
                # create clinical data vector
                (
                    OpConcat(),
                    dict(
                        keys_in=["data.input.gender", "data.input.age_one_hot", "data.metadata.SliceThickness"],
                        key_out="data.input.clinical",
                    ),
                ),
                # fixed image normalization
                (OpToNumpy(), dict(key="data.input.img", dtype=np.float32)),  # cast to float
                (OpClip(), dict(key="data.input.img", clip=clip_range)),
                (OpToRange(), dict(key="data.input.img", from_range=clip_range, to_range=(0.0, 1.0))),
                # Numpy to tensor
                (OpToTensor(), dict(key="data.input.img", dtype=torch.float32)),
                (OpToTensor(), dict(key="data.input.clinical", dtype=torch.float32)),
                # add channel dimension -> [C=1, D, H, W]
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
            ],
        )

        # augmentation
        if train:
            dynamic_pipeline.extend(
                [
                    (OpLambda(partial(torch.squeeze, dim=0)), dict(key="data.input.img")),
                    # affine augmentation - will apply the same affine transformation on each slice
                    (
                        OpRandApply(OpSample(OpAugAffine2D()), 0.5),
                        dict(
                            key="data.input.img",
                            rotate=Uniform(-180.0, 180.0),
                            scale=Uniform(0.8, 1.2),
                            flip=(RandBool(0.5), RandBool(0.5)),
                            translate=(RandInt(-15, 15), RandInt(-15, 15)),
                        ),
                    ),
                    # color augmentation - check if it is useful in CT images
                    # (OpSample(OpAugColor()), dict(
                    #     key="data.input.img",
                    #     gamma=Uniform(0.8,1.2),
                    #     contrast=Uniform(0.9,1.1),
                    #     add=Uniform(-0.01, 0.01)
                    # )),
                    # add channel dimension -> [C=1, D, H, W]
                    (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
                ]
            )

        return dynamic_pipeline

    @staticmethod
    def dataset(
        data_path: str,
        cache_dir: str,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        train: bool = False,
        output_shape: Tuple[int, int, int] = (32, 256, 256),
        clip_range: Tuple[float, float] = (-200, 800),
    ) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param reset_cache: set to True tp reset the cache
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        :param train: True if used for training  - adds augmentation operations to the pipeline
        :param output_shape: fixed shape to resize the image to
        :param clip_range: clip the original voxels values to fit this range
        """

        if sample_ids is None:
            sample_ids = STOIC21.sample_ids(data_path)

        static_pipeline = STOIC21.static_pipeline(data_path, output_shape=output_shape)
        dynamic_pipeline = STOIC21.dynamic_pipeline(train=train, clip_range=clip_range)

        cacher = SamplesCacher(
            f"stoic_cache_ver{STOIC21.STOIC21_DATASET_VER}",
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
