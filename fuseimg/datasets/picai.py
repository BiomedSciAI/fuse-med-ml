from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpNormalizeAgainstSelf
from fuseimg.data.ops.aug.geometry import OpAugAffine2D, OpAugSqueeze3Dto2D, OpAugUnsqueeze3DFrom2D
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D
from fuse.data.ops.ops_common import OpConcat, OpLambda, OpLookup, OpToOneHot
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
import skimage
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
    def static_pipeline(data_dir: str, target: str) -> PipelineDefault:
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
                (OpLambda(partial(skimage.transform.resize,
                                                output_shape=(23, 320, 320),
                                                mode='reflect',
                                                anti_aliasing=True,
                                                preserve_range=True)), dict(key="data.input.img")),
                (OpNormalizeAgainstSelf(), dict(key="data.input.img")),
                (OpToNumpy(), dict(key="data.input.img", dtype=np.float32)),
                
                # (OpResizeAndPad2D(), dict(key="data.input.img", resize_to=(2200, 1200), padding=(60, 60))),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(data_source: pd.DataFrame,train: bool = False, aug_params: NDict = None):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train : True iff we request dataset for train purpouse
        """
        ops = []
        bool_map = {"NO": 0, "YES": 1}
        ops +=[
                (OpToTensor(), dict(key="data.input.img", dtype=torch.float32)),
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
                (
                    OpReadDataframe(
                        data_source,
                        key_column="index",
                        key_name="data.input.img_path",
                        #'psa','psad','prostate_volume','histopath_type','lesion_GS','lesion_ISUP','case_ISUP'
                        columns_to_extract=['index','patient_id','study_id','mri_date','patient_age','case_csPCa','case_ISUP'],
                        rename_columns=dict(
                            patient_id="data.patientID", case_csPCa="data.gt.classification", case_ISUP="data.gt.subtype"
                        ),
                    ),
                    dict(),
                ),
                (OpLookup(bool_map), dict(key_in="data.gt.classification", key_out="data.gt.classification")),
            ]
        if train:
            ops +=[
                    # affine augmentation - will apply the same affine transformation on each slice
                    
                    (OpAugSqueeze3Dto2D(), dict(key='data.input.img', axis_squeeze=1)),
                    (OpRandApply(OpSample(OpAugAffine2D()), aug_params['apply_aug_prob']),
                         dict(key="data.input.img",
                              rotate=Uniform(*aug_params['rotate']),
                              scale=Uniform(*aug_params['scale']),
                              flip=(aug_params['flip'], aug_params['flip']),
                              translate=(RandInt(*aug_params['translate']), RandInt(*aug_params['translate'])))),
                        (OpAugUnsqueeze3DFrom2D(), dict(key='data.input.img', axis_squeeze=1, channels=1)),
                ]
        dynamic_pipeline = PipelineDefault("picai_dynamic", ops)
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
        aug_params= NDict,
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

        static_pipeline = PICAI.static_pipeline(data_dir,target)
        dynamic_pipeline = PICAI.dynamic_pipeline(input_source_gt,train=train,aug_params=aug_params)

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
