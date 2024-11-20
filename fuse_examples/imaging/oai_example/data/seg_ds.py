from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.color import OpToRange, OpClip
from fuseimg.data.ops.color import OpNormalizeAgainstSelf as selfRange
from fuseimg.data.ops.aug.color import OpAugGaussian
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.geometry import (
    OpAugAffine2D,
)
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.ops_read import OpReadDataframe

from fuse.data.ops.ops_common import OpLambda, OpZScoreNorm, OpRepeat
from fuse.utils import NDict
from functools import partial
from typing import Hashable, List, Optional, Sequence, Union
import torch
import pandas as pd
import numpy as np
import os
from fuse.data.utils.sample import get_sample_id

# from fuseimg.data.ops import ops_mri
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


from fuse_examples.imaging.oai_example.data.data_ops import (
    OpNormalizeMRI,
    OpResize3D,
    OpRandomFlip,
    OpSegToOneHot,
    OpRandomCrop,
    OpPickSlice,
)
from volumentations import *


class SegOAI:
    @staticmethod
    def sample_ids(df: pd.DataFrame) -> list:
        return SegOAI.get_existing_sample_ids(df)

    @staticmethod
    def get_existing_sample_ids(df: pd.DataFrame) -> list:
        """
        get all the sample ids that have a zip file in the specified path
        """
        existing_files = df["idx"].values
        return existing_files

    @staticmethod
    def static_pipeline(
        df: pd.DataFrame, im2D: bool = False, drop_blank_slices: bool = False
    ) -> PipelineDefault:
        static_pipeline = PipelineDefault(
            "static",
            [
                # decoding sample ID
                (OpReadDataframe(df, key_column="idx"), dict()),
                (OpLoadImage(""), dict(key_in="img_path", key_out="img", format="nib")),
                (OpLoadImage(""), dict(key_in="seg_path", key_out="seg", format="nib")),
                (OpNormalizeMRI(), dict(key="img", to_range=[0, 1])),
            ],
        )
        if im2D:
            static_pipeline.extend(
                [
                    (OpPickSlice("img", "seg", "slice", drop_blank_slices), dict()),
                ]
            )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        validation: bool = False,
        resize_to: Sequence = [40, 224, 224],
        num_classes: int = 7,
    ):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train : True iff we request dataset for train purpouse
        """

        dynamic_pipeline = PipelineDefault("dynamic", [])
        # augmentation

        if validation:
            if resize_to != None:
                if len(resize_to) == 1:
                    print(f"resize take every {resize_to} on the depth dim")
                    dynamic_pipeline.extend(
                        [
                            (
                                OpLambda(lambda x: x[:: resize_to[0], :, :]),
                                dict(key="img"),
                            ),
                            (
                                OpLambda(lambda x: x[:: resize_to[0], :, :]),
                                dict(key="seg"),
                            ),
                        ]
                    )
                else:
                    dynamic_pipeline.extend(
                        [
                            (OpResize3D(), dict(key="img", shape=resize_to)),
                            (
                                OpResize3D(),
                                dict(key="seg", shape=resize_to, segmentation=True),
                            ),
                        ]
                    )
            else:
                print("no Resize")
            dynamic_pipeline.extend(
                [
                    (OpToTensor(), dict(key="img", dtype=torch.float32)),
                    (OpToTensor(), dict(key="seg", dtype=torch.int8)),
                ]
            )
        else:
            if resize_to != None:
                if len(resize_to) == 1:
                    starting_idx = np.random.randint(0, 4)
                    dynamic_pipeline.extend(
                        [
                            (
                                OpLambda(
                                    lambda x: x[starting_idx :: resize_to[0], :, :]
                                ),
                                dict(key="img"),
                            ),
                            (
                                OpLambda(
                                    lambda x: x[starting_idx :: resize_to[0], :, :]
                                ),
                                dict(key="seg"),
                            ),
                        ]
                    )
                else:
                    dynamic_pipeline.extend(
                        [
                            (
                                OpRandomCrop(),
                                dict(key=["img", "seg"], scale=[0.7, 1.0]),
                            ),
                            (OpResize3D(), dict(key="img", shape=resize_to)),
                            (
                                OpResize3D(),
                                dict(key="seg", shape=resize_to, segmentation=True),
                            ),
                        ]
                    )
            dynamic_pipeline.extend(
                [
                    (OpRandomFlip(), dict(key=["img", "seg"])),
                    (OpToTensor(), dict(key="img", dtype=torch.float32)),
                    (OpToTensor(), dict(key="seg", dtype=torch.int8)),
                    (
                        OpRandApply(OpSample(OpAugAffine2D()), 0.5),
                        dict(
                            key="img",
                            scale=Uniform(0.8, 1.2),
                        ),
                    ),
                    # (OpRandApply(OpAugGaussian(),0.1), dict(key="img"))
                ]
            )
        dynamic_pipeline.extend(
            [
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="img")),
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="seg")),
                (OpSegToOneHot(n_classes=num_classes), dict(key="seg")),
            ]
        )

        return dynamic_pipeline

    @staticmethod
    def dataset(
        csv_path: Union[str, pd.DataFrame],
        cache_dir: str = None,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        resize_to: tuple = (40, 224, 224),
        validation: bool = False,
        num_classes: int = 7,
        im2D: bool = False,
        drop_blank_slices: bool = False,
    ):
        """
        Creates Fuse Dataset single object (either for training, validation and test or user defined set)

        :param data_dir:                    dataset root path
        :param series_config                configuration of the selected series from the ukbb zip
        :param input_source_gt              dataframe with ground truth file
        :param cache_dir:                   Optional, name of the cache folder
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        :param train: True if used for training  - adds augmentation operations to the pipeline
        :return: DatasetDefault object
        """
        if isinstance(csv_path, str):
            df = pd.read_csv(csv_path)
        else:
            df = csv_path

        if sample_ids is None:
            sample_ids = SegOAI.sample_ids(df)

        static_pipeline = SegOAI.static_pipeline(
            df, im2D=im2D, drop_blank_slices=drop_blank_slices
        )
        dynamic_pipeline = SegOAI.dynamic_pipeline(
            validation=validation, resize_to=resize_to, num_classes=num_classes
        )

        if cache_dir is not None:
            cacher = SamplesCacher(
                "oai",
                static_pipeline,
                cache_dirs=[cache_dir],
                restart_cache=reset_cache,
                audit_first_sample=False,
                audit_rate=None,
                workers=num_workers,
                use_pipeline_hash=False,
            )

            my_dataset = DatasetDefault(
                sample_ids=sample_ids,
                static_pipeline=static_pipeline,
                dynamic_pipeline=dynamic_pipeline,
                cacher=cacher,
            )
        else:
            my_dataset = DatasetDefault(
                sample_ids=sample_ids,
                static_pipeline=static_pipeline,
                dynamic_pipeline=dynamic_pipeline,
                cacher=None,
                allow_uncached_sample_morphing=False,
            )

        my_dataset.create(num_workers=num_workers)
        return my_dataset
