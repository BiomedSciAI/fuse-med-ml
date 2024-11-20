from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.color import OpToRange, OpClip
from fuseimg.data.ops.color import OpNormalizeAgainstSelf as selfRange
from fuseimg.data.ops.aug.color import OpAugGaussian
from fuseimg.data.ops.aug.geometry import (
    OpAugAffine2D,
    OpRotation3D,
    OpResizeTo,
    OpCrop3D,
)
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.ops_read import OpReadDataframe
from torchvision import transforms

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
    OpLoadData,
    OpNormalizeMRI,
    OpResize3D,
    OpVolumentation,
    OpMask3D,
    OpDinoCrops,
    OpRandomCrop,
    OpRandomFlip,
)
from volumentations import Compose, GaussianNoise, Flip, Rotate


class OAI:
    @staticmethod
    def sample_ids(df):
        return OAI.get_existing_sample_ids(df)

    @staticmethod
    def get_existing_sample_ids(df):
        """
        get all the sample ids that have a zip file in the specified path
        """
        existing_files = df["accession_number"].values
        return existing_files

    @staticmethod
    def static_pipeline(
        df: pd.DataFrame, im2D=False
    ) -> PipelineDefault:
        max_key = "max_val" if im2D else None
        static_pipeline = PipelineDefault(
            "static",
            [
                # decoding sample ID
                (
                    OpReadDataframe(
                        df,
                        key_column="accession_number",
                    ),
                    dict(),
                ),
                (OpLoadData(path_key="path", im2D=im2D), dict()),
                (OpNormalizeMRI(), dict(key="img", to_range=[0, 1], max_key=max_key)),
                #
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        n_crops=2,
        mae_cfg=None,
        for_classification: bool = True,
        validation: bool = False,
        resize_to: Sequence = [40, 224, 224],
        im2D=False,
    ):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train : True iff we request dataset for train purpouse
        """
        if for_classification | (mae_cfg is not None):
            repeat_for = [dict(key="img")]
        else:
            repeat_for = [dict(key=f"crop_{i}") for i in range(n_crops)]

        dynamic_pipeline = PipelineDefault("dynamic", [])
        # augmentation

        if for_classification:
            if validation:
                dynamic_pipeline.extend(
                    [
                        (OpResize3D(), dict(key="img", shape=resize_to)),
                        (OpRepeat(OpToTensor(), repeat_for), dict(dtype=torch.float32)),
                        (
                            OpRepeat(
                                OpLambda(partial(torch.unsqueeze, dim=0)), repeat_for
                            ),
                            dict(),
                        ),
                    ]
                )
            else:
                dynamic_pipeline.extend(
                    [
                        (OpRandomCrop(), dict(key=["img"], scale=[0.7, 1.0])),
                        (OpResize3D(), dict(key="img", shape=resize_to)),
                        (OpToTensor(), dict(key="img", dtype=torch.float)),
                        (OpRandomFlip(), dict(key=["img"])),
                        (
                            OpRandApply(OpSample(OpAugAffine2D()), 0.5),
                            dict(
                                key="img",
                                scale=Uniform(0.9, 1.1),
                                translate=(RandInt(-10, 10), RandInt(-10, 10)),
                            ),
                        ),
                        (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="img")),
                    ]
                )
        elif mae_cfg is not None:
            if im2D:
                dynamic_pipeline.extend(
                    [
                        (OpRandomFlip(), dict(key="img")),
                        (
                            OpMask3D(
                                mask_percentage=mae_cfg["mask_percentage"],
                                cuboid_size=mae_cfg["cuboid_size"],
                            ),
                            dict(key="img", out_key="masked_img"),
                        ),
                        (OpResizeTo(False), dict(key="img", output_shape=[384, 384])),
                        (
                            OpResizeTo(False),
                            dict(key="masked_img", output_shape=[384, 384]),
                        ),
                        (OpToTensor(), dict(key="img", dtype=torch.float)),
                        (OpToTensor(), dict(key="masked_img", dtype=torch.float)),
                        (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="img")),
                        (
                            OpLambda(partial(torch.unsqueeze, dim=0)),
                            dict(key="masked_img"),
                        ),
                    ]
                )
            else:
                dynamic_pipeline.extend(
                    [
                        (OpRandomFlip(), dict(key="img")),
                        (OpResize3D(), dict(key="img", shape=resize_to)),
                        (
                            OpMask3D(
                                mask_percentage=mae_cfg["mask_percentage"],
                                cuboid_size=mae_cfg["cuboid_size"],
                            ),
                            dict(key="img", out_key="masked_img"),
                        ),
                        (OpToTensor(), dict(key="img", dtype=torch.float)),
                        (OpToTensor(), dict(key="masked_img", dtype=torch.float)),
                        (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="img")),
                        (
                            OpLambda(partial(torch.unsqueeze, dim=0)),
                            dict(key="masked_img"),
                        ),
                    ]
                )
        else:  # dino processing
            dynamic_pipeline.extend(
                [
                    (OpDinoCrops(), dict(key="img", n_crops=n_crops)),
                ]
            )
            if im2D:
                dynamic_pipeline.extend(
                    [
                        (OpRepeat(OpRandomFlip(), repeat_for), dict()),
                        (OpRepeat(OpToTensor(), repeat_for), dict(dtype=torch.float32)),
                        (
                            OpRepeat(
                                OpRandApply(OpSample(OpAugAffine2D()), 0.5), repeat_for
                            ),
                            dict(
                                rotate=Uniform(-15.0, 15.0),
                                scale=Uniform(0.8, 1.2),
                                translate=(RandInt(-15, 15), RandInt(-15, 15)),
                            ),
                        ),
                        (
                            OpRepeat(OpRandomCrop(), repeat_for[:2]),
                            dict(scale=(0.8, 1.0), on_depth=False),
                        ),
                        (
                            OpRepeat(OpRandomCrop(), repeat_for[2:]),
                            dict(scale=(0.1, 0.5), on_depth=False),
                        ),
                        # (OpRepeat(OpLambda(transforms.Compose([
                        #                     transforms.RandomHorizontalFlip(p=0.5),
                        #                     # transforms.RandomApply([
                        #                     #     transforms.GaussianBlur((3, 3), (1.0, 2.0))
                        #                     # ], p=0.1),
                        #                     # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                        #                     transforms.RandomAutocontrast(p=0.5),]),
                        #                     ), repeat_for), dict()),
                        (OpRepeat(OpToNumpy(), repeat_for), dict()),
                        (
                            OpRepeat(OpResizeTo(False), repeat_for[:2]),
                            dict(output_shape=[224, 224]),
                        ),
                        (
                            OpRepeat(OpResizeTo(False), repeat_for[2:]),
                            dict(output_shape=[128, 128]),
                        ),
                        (OpRepeat(OpToTensor(), repeat_for), dict(dtype=torch.float32)),
                        (
                            OpRepeat(
                                OpLambda(partial(torch.unsqueeze, dim=0)), repeat_for
                            ),
                            dict(),
                        ),
                    ]
                )

            else:
                dynamic_pipeline.extend(
                    [
                        (
                            OpRepeat(OpRandomCrop(), repeat_for[:2]),
                            dict(scale=(0.8, 1.0), on_depth=False),
                        ),
                        (
                            OpRepeat(OpRandomCrop(), repeat_for[2:]),
                            dict(scale=(0.1, 0.5), on_depth=False),
                        ),
                        (
                            OpRepeat(
                                OpVolumentation(
                                    Compose(
                                        [
                                            Rotate(
                                                (-15, 15), (-15, 15), (-15, 15), p=0.5
                                            ),
                                            Flip(0, p=0.5),
                                            Flip(1, p=0.5),
                                            Flip(2, p=0.5),
                                            # ColorJitter3D(brightness=0.4, contrast=0.4, saturation=1, hue=0,p=0.5),
                                            # RandomRotate90((1, 2), p=0.5),
                                        ]
                                    )
                                ),
                                repeat_for,
                            ),
                            dict(),
                        ),
                        # Global Gaussian 1
                        (
                            OpVolumentation(
                                Compose([GaussianNoise(var_limit=(0.001, 0.1), p=0.6)])
                            ),
                            dict(key="crop_0"),
                        ),
                        # Global Gaussian 2 + solarization
                        (
                            OpVolumentation(
                                Compose(
                                    [
                                        GaussianNoise(var_limit=(0.001, 0.01), p=0.1)
                                        # RandomSolarize(p=0.2)
                                    ]
                                )
                            ),
                            dict(key="crop_1"),
                        ),
                        # Local Gaussian
                        (
                            OpRepeat(
                                OpVolumentation(
                                    Compose(
                                        [GaussianNoise(var_limit=(0.001, 0.005), p=0.5)]
                                    )
                                ),
                                repeat_for[2:],
                            ),
                            dict(),
                        ),
                        (
                            OpRepeat(OpResize3D(), repeat_for[:2]),
                            dict(shape=[40, 224, 224]),
                        ),
                        (
                            OpRepeat(OpResize3D(), repeat_for[2:]),
                            dict(shape=[40, 128, 128]),
                        ),
                        (OpRepeat(OpToTensor(), repeat_for), dict(dtype=torch.float32)),
                        (
                            OpRepeat(
                                OpLambda(partial(torch.unsqueeze, dim=0)), repeat_for
                            ),
                            dict(),
                        ),
                        # (OpRepeat(OpResizeTo(True), repeat_for), dict(output_shape=resize_to)),
                        # (OpRepeat(OpToTensor(), repeat_for), dict(dtype=torch.float32)),
                    ]
                )

        return dynamic_pipeline

    @staticmethod
    def dataset(
        csv_path: Union[str, pd.DataFrame],
        cache_dir: str = None,
        reset_cache: bool = True,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        resize_to: tuple = (40, 224, 224),
        for_classification: bool = True,
        validation: bool = False,
        n_crops: int = 4,
        mae_cfg: dict = None,
        im2D: bool = False,
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
            sample_ids = OAI.sample_ids(df)

        static_pipeline = OAI.static_pipeline(df, im2D)
        dynamic_pipeline = OAI.dynamic_pipeline(
            mae_cfg=mae_cfg,
            for_classification=for_classification,
            validation=validation,
            n_crops=n_crops,
            resize_to=resize_to,
            im2D=im2D,
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
                # dynamic_pipeline=dynamic_pipeline,
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

        my_dataset.create()  # num_workers = num_workers)
        return my_dataset
