import os
from zipfile import ZipFile
from fuse.utils.file_io.file_io import create_dir
import wget
from typing import Hashable, Optional, Sequence, List, Tuple
import torch

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLambda, OpOverrideNaN
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
        if int(sample_dict[f"data.cls_labels.{cls_name}"]) == 1:
            label = idx

    sample_dict["data.label"] = label
    return sample_dict


class ISIC:
    """
    ISIC 2019 challenge to classify dermoscopic images and clinical data among nine different diagnostic categories.
    """

    # bump whenever the static pipeline modified
    DATASET_VER = 0

    CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    @staticmethod
    def download(data_path: str, sample_ids_to_download: Optional[Sequence[str]] = None) -> None:
        """
        Download images and metadata from ISIC challenge.
        Doesn't download again if data exists.
        :param data_path: location to store the data
        :param sample_ids_to_downlad: use if you want to download subset of the data. None will download and extract all the data.
        """
        create_dir(data_path)
        create_dir(os.path.join(data_path, "ISIC2019"))
        path = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Input")
        if not os.path.exists(path):
            print("\nExtract ISIC-2019 training input ... (this may take a few minutes)")

            url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
            filename = os.path.join(data_path, "ISIC_2019_Training_Input.zip")
            print(f"Extract to {path}")
            if not os.path.exists(filename):
                wget.download(url, os.path.join(data_path, "ISIC_2019_Training_Input.zip"))

            if sample_ids_to_download is not None:
                members = [os.path.join("ISIC_2019_Training_Input", m + ".jpg") for m in sample_ids_to_download]
            else:
                members = None

            with ZipFile(filename, "r") as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(path=os.path.join(data_path, "ISIC2019"), members=members)

            print("Extracting ISIC-2019 training input: done")

        path = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_GroundTruth.csv")

        if not os.path.exists(path):
            print("\nExtract ISIC-2019 training gt ... (this may take a few minutes)")

            url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
            wget.download(url, os.path.join(data_path, "ISIC2019/ISIC_2019_Training_GroundTruth.csv"))

            print("Extracting ISIC-2019 training gt: done")

        path = os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Metadata.csv")

        if not os.path.exists(path):
            print("\nExtract ISIC-2019 metadata ... (this may take a few minutes)")

            url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
            wget.download(url, os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Metadata.csv"))

            print("Extracting ISIC-2019 metadata: done")

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
        rename_cls_labels = {c: f"data.cls_labels.{c}" for c in ISIC.CLASS_NAMES}
        rename_cls_labels["image"] = "data.cls_labels.sample_id"  # also extract image (sample_id)
        rename_metadata = {
            "age_approx": "data.input.clinical.age_approx",
            "anatom_site_general": "data.input.clinical.anatom_site_general",
            "sex": "data.input.clinical.sex",
            "image": "data.input.clinical.sample_id",
        }

        static_pipeline = PipelineDefault(
            "static",
            [
                # Decoding sample ID
                (OpISICSampleIDDecode(), dict()),
                # Load Image
                (
                    OpLoadImage(os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Input")),
                    dict(key_in="data.input.img_path", key_out="data.input.img"),
                ),
                # Normalize Images to range [0, 1]
                (OpToRange(), dict(key="data.input.img", from_range=(0, 255), to_range=(0, 1))),
                # Read labels into sample_dict. Each class will have a different entry.
                (
                    OpReadDataframe(
                        data_filename=os.path.join(data_path, "ISIC2019/ISIC_2019_Training_GroundTruth.csv"),
                        key_column="data.cls_labels.sample_id",
                        columns_to_extract=list(rename_cls_labels.keys()),
                        rename_columns=rename_cls_labels,
                    ),
                    dict(),
                ),
                # Read metadata into sample_dict
                (
                    OpReadDataframe(
                        data_filename=os.path.join(data_path, "ISIC2019/ISIC_2019_Training_Metadata.csv"),
                        key_column="data.input.clinical.sample_id",
                        columns_to_extract=list(rename_metadata.keys()),
                        rename_columns=rename_metadata,
                    ),
                    dict(),
                ),
                (OpOverrideNaN(), dict(key="data.input.clinical.anatom_site_general", value_to_fill="N/A")),
                (OpOverrideNaN(), dict(key="data.input.clinical.sex", value_to_fill="N/A")),
                (OpOverrideNaN(), dict(key="data.input.clinical.age_approx", value_to_fill=-1.0)),
                # Squeeze labels into sample_dict['data.label']
                (OpLambda(func=derive_label), dict()),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        train: bool = False, append: Optional[Sequence[Tuple[OpBase, dict]]] = None
    ) -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train: add augmentation if True
        :param append: pipeline steps to append at the end of the suggested pipeline
        """

        dynamic_pipeline = [
            # Resize images to 300x300x3
            (
                OpResizeTo(channels_first=True),
                dict(key="data.input.img", output_shape=(300, 300, 3), mode="reflect", anti_aliasing=True),
            ),
            # Convert to tensor for the augmentation process
            (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
        ]

        if train:
            dynamic_pipeline += [
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
            ]

        if append is not None:
            dynamic_pipeline += append

        return PipelineDefault("dynamic", dynamic_pipeline)

    @staticmethod
    def dataset(
        data_path: str,
        cache_path: str,
        train: bool = False,
        reset_cache: bool = False,
        num_workers: int = 10,
        append_dyn_pipeline: Optional[Sequence[Tuple[OpBase, dict]]] = None,
        samples_ids: Optional[Sequence[Hashable]] = None,
    ) -> DatasetDefault:
        """
        Get cached dataset
        :param train: if true returns the train dataset, else the validation one.
        :param reset_cache: set to True to reset the cache
        :param num_workers: number of processes used for caching
        :param append_dyn_pipeline: pipeline steps to append at the end of the suggested dynamic pipeline
        :param sample_ids: dataset including the specified sample_ids or None for all the samples.
        """
        # Download data if doesn't exist
        ISIC.download(data_path=data_path, sample_ids_to_download=samples_ids)

        if samples_ids is None:
            samples_ids = ISIC.sample_ids(data_path)

        static_pipeline = ISIC.static_pipeline(data_path)
        dynamic_pipeline = ISIC.dynamic_pipeline(train, append=append_dyn_pipeline)

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
