from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.image_loader import OpLoadImage
from fuse.data.ops.ops_read import OpReadDataframe

from functools import partial
from typing import Hashable, Optional, Sequence, Union, Tuple
import torch
import pandas as pd
from fuse_examples.imaging.oai_example.data.data_ops import (
    OpNormalizeMRI,
    OpResize3D,
    OpRandomFlip,
    OpSegToOneHot,
    OpRandomCrop,
)


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
    def static_pipeline(df: pd.DataFrame) -> PipelineDefault:
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
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        validation: bool = False, resize_to: Tuple[int, int, int] = [40, 224, 224]
    ) -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train : True iff we request dataset for train purpouse
        """

        dynamic_pipeline = PipelineDefault("dynamic", [])
        # augmentation

        if validation:
            if resize_to is not None:
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
        else:
            dynamic_pipeline.extend(
                [
                    (OpRandomCrop(), dict(key=["img", "seg"], scale=[0.7, 1.0])),
                    (OpResize3D(), dict(key="img", shape=resize_to)),
                    (OpResize3D(), dict(key="seg", shape=resize_to, segmentation=True)),
                    (OpRandomFlip(), dict(key=["img", "seg"])),
                ]
            )
        dynamic_pipeline.extend(
            [
                (OpToTensor(), dict(key="img", dtype=torch.float32)),
                (OpToTensor(), dict(key="seg", dtype=torch.int8)),
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="img")),
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="seg")),
                (OpSegToOneHot(n_classes=7), dict(key="seg")),
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
        validation: bool = False,
    ) -> DatasetDefault:
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

        static_pipeline = SegOAI.static_pipeline(df)
        dynamic_pipeline = SegOAI.dynamic_pipeline(
            validation=validation, resize_to=resize_to
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
