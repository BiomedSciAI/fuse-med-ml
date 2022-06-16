import os
from zipfile import ZipFile
import wget
import logging
from typing import Hashable, Optional, Sequence, List
import torch

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.color import OpToRange

from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


class OpISICSampleIDDecode(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id into image file name
        """
        sid = get_sample_id(sample_dict)

        img_filename_key = "data.input.img_path"
        sample_dict[img_filename_key] = sid + ".jpg"

        return sample_dict


def derive_label(sample_dict: NDict) -> NDict:
    """
    Takes the sample's ndict with the labels as key:value and assigns to sample_dict['data.label'] the index of the sample's class.
    Also delete all the labels' keys from sample_dict.

    for example:
        If the sample contains {'MEL': 0, 'NV': 1, 'BCC': 0, 'AK': 0, ... }
        will assign, sample_dict['data.label'] = 1 ('NV's index).
        Afterwards the sample_dict won't contain the class' names & values.
    """
    classes_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    label = 0
    for idx, cls_name in enumerate(classes_names):
        if int(sample_dict[cls_name]) == 1:
            label = idx

        del sample_dict[cls_name]

    sample_dict["data.label"] = label
    return sample_dict


class ISIC:
    """
    ISIC 2019 challenge to classify dermoscopic images and clinical data among nine different diagnostic categories.
    """

    # bump whenever the static pipeline modified
    DATASET_VER = 0

    @staticmethod
    def download(data_path: str) -> None:
        """
        Download images and metadata from ISIC challenge.
        Doesn't download again if data exists.

        """
        lgr = logging.getLogger("Fuse")

        path = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Input")
        print(f"Training Input Path: {os.path.abspath(path)}")
        if not os.path.exists(path):
            lgr.info("\nExtract ISIC-2019 training input ... (this may take a few minutes)")

            url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
            wget.download(url, ".")

            with ZipFile("ISIC_2019_Training_Input.zip", "r") as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(path=os.path.join(data_path, "ISIC2019"))

            lgr.info("Extracting ISIC-2019 training input: done")

        path = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_GroundTruth.csv")

        if not os.path.exists(path):
            lgr.info("\nExtract ISIC-2019 training gt ... (this may take a few minutes)")

            url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
            wget.download(url, path)

            lgr.info("Extracting ISIC-2019 training gt: done")

    @staticmethod
    def sample_ids(data_path: str) -> List[str]:
        """
        Gets the samples ids in trainset.
        """
        images_path = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Input")

        samples = [f.split(".")[0] for f in os.listdir(images_path) if f.split(".")[-1] == "jpg"]
        return samples

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:

        static_pipeline = PipelineDefault(
            "static",
            [
                # Decoding sample ID
                (OpISICSampleIDDecode(), dict()),
                # Load Image
                (OpLoadImage(data_path), dict(key_in="data.input.img_path", key_out="data.input.img")),
                # Normalize Images to range [0, 1]
                (OpToRange(), dict(key="data.input.img", from_range=(0, 255), to_range=(0, 1))),
                # Read labels into sample_dict. Each class will have a different entry.
                (
                    OpReadDataframe(
                        data_filename=os.path.join(data_path, "../ISIC_2019_Training_GroundTruth.csv"),
                        key_column="image",
                    ),
                    dict(),
                ),
                # Squeeze labels into sample_dict['data.label']
                (OpLambda(func=derive_label), dict()),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(train: bool = False) -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        """

        dynamic_pipeline = PipelineDefault(
            "dynamic",
            [
                # Resize images to 300x300x3
                (
                    OpResizeTo(channels_first=True),
                    dict(key="data.input.img", output_shape=(300, 300, 3), mode="reflect", anti_aliasing=True),
                ),
                # Convert to tensor for the augmentation process
                (OpToTensor(), dict(key="data.input.img")),
                # Augmentation
                (
                    OpSample(OpAugAffine2D()),
                    dict(
                        key="data.input.img",
                        rotate=Uniform(-180.0, 180.0),
                        scale=Uniform(0.9, 1.1),
                        flip=(RandBool(0.3), RandBool(0.3)),
                        translate=(RandInt(-50, 50), RandInt(-50, 50)),
                    ),
                ),
                # Color augmentation
                (
                    OpSample(OpAugColor()),
                    dict(
                        key="data.input.img",
                        gamma=Uniform(0.9, 1.1),
                        contrast=Uniform(0.85, 1.15),
                        add=Uniform(-0.06, 0.06),
                        mul=Uniform(0.95, 1.05),
                    ),
                ),
                # Add Gaussian noise
                (OpAugGaussian(), dict(key="data.input.img", std=0.03)),
                # Convert to float so the tensor will have the same dtype as model
                (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
            ],
        )

        return dynamic_pipeline

    @staticmethod
    def dataset(
        data_path: str,
        cache_path: str,
        train: bool = False,
        reset_cache: bool = False,
        num_workers: int = 10,
        samples_ids: Optional[Sequence[Hashable]] = None,
    ) -> DatasetDefault:
        """
        Get cached dataset
        :param train: if true returns the train dataset, else the validation one.
        :param reset_cache: set to True to reset the cache
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples.
        :param override_partition: If True, will make a new partition of train-validation sets.
        """
        # Download data if doesn't exist
        ISIC.download(data_path=data_path)

        data_dir = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Input")

        if samples_ids is None:
            samples_ids = ISIC.sample_ids(data_path)

        static_pipeline = ISIC.static_pipeline(data_dir)
        dynamic_pipeline = ISIC.dynamic_pipeline(train)

        cacher = SamplesCacher(
            f"isic_cache_ver{ISIC.DATASET_VER}",
            static_pipeline,
            [cache_path],
            restart_cache=reset_cache,
            workers=num_workers,
        )

        my_dataset = DatasetDefault(
            sample_ids=samples_ids, static_pipeline=static_pipeline, dynamic_pipeline=dynamic_pipeline, cacher=cacher
        )

        my_dataset.create()
        return my_dataset
