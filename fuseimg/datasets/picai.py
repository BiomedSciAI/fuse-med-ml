from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpNormalizeAgainstSelf
from fuseimg.data.ops.shape_ops import OpFlipBrightSideOnLeft2D, OpFindBiggestNonEmptyBbox2D, OpResizeAndPad2D
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
from functools import partial
from typing import Hashable, Optional, Sequence
import torch
import pandas as pd
import numpy as np
import pydicom
import os
import glob
from pathlib import Path
from fuse.data.utils.sample import get_sample_id
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


class OpPICAISampleIDDecode(OpBase):
    """
    decodes sample id into image and segmentation filename
    """

    def __call__(self, sample_dict: NDict) -> NDict:
        """ """
        sid = get_sample_id(sample_dict)

        img_filename_key = "data.input.img_path"
        sample_dict[img_filename_key] = sid

        return sample_dict


class PICAI:
    """
    """
    @staticmethod
    def static_pipeline(data_dir: str, data_source: pd.DataFrame, target: str) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault(
            "cmmd_static",
            [
                # decoding sample ID
                (OpPICAISampleIDDecode(), dict()),  # will save image and seg path to "data.input.img_path"
                (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="mha")),
                (OpToNumpy(), dict(key="data.input.img", dtype=np.float32)),
                (OpNormalizeAgainstSelf(), dict(key="data.input.img")),
                # (OpResizeAndPad2D(), dict(key="data.input.img", resize_to=(2200, 1200), padding=(60, 60))),
                (
                    OpReadDataframe(
                        data_source,
                        key_column="index",
                        key_name="data.input.img_path",
                        columns_to_extract=['index','patient_id','study_id','mri_date','patient_age','psa','psad','prostate_volume','histopath_type','lesion_GS','lesion_ISUP','case_ISUP','case_csPCa'],
                        rename_columns=dict(
                            patient_id="data.patientID", case_csPCa="data.gt.classification", case_ISUP="data.gt.subtype"
                        ),
                    ),
                    dict(),
                ),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(train: bool = False):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train : True iff we request dataset for train purpouse
        """
        dynamic_pipeline = PipelineDefault(
            "cmmd_dynamic",
            [
                (OpToTensor(), dict(key="data.input.img", dtype=torch.float32)),
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
            ],
        )
        if train:
            dynamic_pipeline.extend(
                [
                    (
                        OpRandApply(OpSample(OpAugAffine2D()), 0.5),
                        dict(
                            key="data.input.img",
                            rotate=Uniform(-30.0, 30.0),
                            scale=Uniform(0.9, 1.1),
                            flip=(RandBool(0.3), RandBool(0.5)),
                            translate=(RandInt(-10, 10), RandInt(-10, 10)),
                        ),
                    ),
                    (
                        OpRandApply(OpSample(OpAugColor()), 0.5),
                        dict(
                            key="data.input.img",
                            gamma=Uniform(0.9, 1.1),
                            contrast=Uniform(0.85, 1.15),
                            mul=Uniform(0.95, 1.05),
                            add=Uniform(-0.06, 0.06),
                        ),
                    ),
                ]
            )
        return dynamic_pipeline


    @staticmethod
    def dataset(
        data_dir: str,
        clinical_file: str,
        target: str,
        cache_dir: str = None,
        reset_cache: bool = True,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        train: bool = False,
    ):
        """
        Creates Fuse Dataset single object (either for training, validation and test or user defined set)
        :param data_dir:                    dataset root path
        :param clinical_file                path to clinical_file
        :param target                       target name used from the ground truth dataframe
        :param cache_dir:                   Optional, name of the cache folder
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        :param train: True if used for training  - adds augmentation operations to the pipeline
        :return: DatasetDefault object
        """

        input_source_gt = pd.read_csv(clinical_file)
        input_source_gt['index'] = input_source_gt['patient_id'].astype(str)+"/"+input_source_gt['patient_id'].astype(str)+"_"+input_source_gt['study_id'].astype(str)+"_t2w.mha"
        all_sample_ids = input_source_gt['index'].to_list()

        if sample_ids is None:
            sample_ids = all_sample_ids

        static_pipeline = PICAI.static_pipeline(data_dir, input_source_gt, target)
        dynamic_pipeline = PICAI.dynamic_pipeline(train=train)

        cacher = SamplesCacher(
            "cache_ver",
            static_pipeline,
            cache_dirs=[cache_dir],
            restart_cache=reset_cache,
            audit_first_sample=False,
            audit_rate=None,
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
