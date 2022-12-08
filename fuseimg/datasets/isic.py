import os
from zipfile import ZipFile
from fuse.utils.file_io.file_io import create_dir
import wget
from typing import Hashable, Optional, Sequence, List, Tuple
import torch
import numpy as np
import pytorch_lightning as pl
from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLambda, OpOverrideNaN, OpConcat
from fuseimg.data.ops.color import OpToRange
from fuse.data.ops.ops_aug_tabular import OpAugOneHot
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
from fuse.data.utils.samplers import BatchSamplerDefault
from torch.utils.data import DataLoader
from fuse.data.utils.collates import CollateDefault


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
    classes_names = ISIC.CLASS_NAMES

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
    SEX_INDEX = {"male": 0, "female": 1, "N/A": 2}
    ANATOM_SITE_INDEX = {
        "anterior torso": 0,
        "upper extremity": 1,
        "posterior torso": 2,
        "lower extremity": 3,
        "lateral torso": 4,
        "head/neck": 5,
        "palms/soles": 6,
        "oral/genital": 7,
        "N/A": 8,
    }

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
                # Encode meta-data
                (
                    OpEncodeMetaData(),
                    dict(
                        key_site="data.input.clinical.anatom_site_general",
                        key_sex="data.input.clinical.sex",
                        key_age="data.input.clinical.age_approx",
                        out_prefix="data.input.clinical.encoding",
                    ),
                ),
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
                # Image Augmentation
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
                # Meta-data Augmentation
                # Drop age with prob
                (
                    OpRandApply(OpAugOneHot(), probability=0.05),
                    dict(key="data.input.clinical.encoding.sex", idx=2, mode="default"),
                ),
                # switch age class with prob, except from unknowns
                (
                    OpRandApply(OpAugOneHot(), probability=0.05),
                    dict(key="data.input.clinical.encoding.age", mode="ranking", freeze_indices=[6]),
                ),
            ]

        final = [
            # concat tabular data to one vector and cast it to tensor
            (
                OpConcat(),
                dict(
                    keys_in=[f"data.input.clinical.encoding.{c}" for c in ["site", "sex", "age"]],
                    key_out="data.input.clinical.all",
                ),
            ),
            (OpToTensor(), dict(key="data.input.clinical.all", dtype=torch.float)),
        ]

        if append is not None:
            dynamic_pipeline += append

        dynamic_pipeline += final

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
        :param train: if true, apply augmentations in the dynamic pipeline
        :param reset_cache: set to True to reset the cache
        :param num_workers: number of processes used for caching
        :param append_dyn_pipeline: pipeline steps to append at the end of the suggested dynamic pipeline
        :param samples_ids: dataset including the specified samples_ids or None for all the samples.
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


class ISICDataModule(pl.LightningDataModule):
    """
    Example of a custom Lightning datamodule that use FuseMedML tools.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        num_workers: int,
        batch_size: int,
        train_folds: List[int],
        validation_folds: List[int],
        infer_folds: List[int],
        split_filename: str,
        sample_ids: Optional[Sequence[Hashable]] = None,
        reset_cache: bool = False,
        reset_split: bool = False,
        use_batch_sampler: bool = True,
    ):
        """
        :param data_dir: path to data directory
        :param cache_dir: path to cache directory
        :param num_workers: number of process to use
        :param batch_size: model's batch size
        :param train_folds: which folds will be used for training
        :param validation_folds: which folds will be used for validation
        :param infer_folds: which folds will be used for inference (final evaluation)
        :param split_filename: path to file that will contain the data's split to folds
        :param sample_ids: subset of the sample ids to use, if None - use all.
        :param reset_cache: set True to reset the cache data
        :param reset_split: set True to reset the split file
        :param use_batch_sample: set True to use Fuse's custom balanced batch sampler (see BatchSamplerDefault class)
        """
        super().__init__()
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._use_batch_sampler = use_batch_sampler

        # divide into balanced train, validation and evaluation folds
        self._train_ids = []
        self._validation_ids = []
        self._infer_ids = []

        all_data = ISIC.dataset(
            self._data_dir,
            self._cache_dir,
            num_workers=self._num_workers,
            train=False,
            reset_cache=reset_cache,
            samples_ids=sample_ids,
        )

        folds = dataset_balanced_division_to_folds(
            dataset=all_data,
            keys_to_balance=["data.label"],
            nfolds=len(train_folds + validation_folds + infer_folds),
            output_split_filename=split_filename,
            reset_split=reset_split,
        )

        for fold in train_folds:
            self._train_ids += folds[fold]

        for fold in validation_folds:
            self._validation_ids += folds[fold]

        for fold in infer_folds:
            self._infer_ids += folds[fold]

    def setup(self, stage: str) -> None:
        """
        creates datasets by stage
        called on every process in DDP

        :param stage: trainer stage
        """
        # assign train/val datasets
        if stage == "fit":
            self._train_dataset = ISIC.dataset(
                self._data_dir, self._cache_dir, num_workers=self._num_workers, train=True, samples_ids=self._train_ids
            )
            self._validation_dataset = ISIC.dataset(
                self._data_dir,
                self._cache_dir,
                num_workers=self._num_workers,
                train=False,
                samples_ids=self._validation_ids,
            )

        # assign prediction (infer) dataset
        if stage == "predict":
            self._predict_dataset = ISIC.dataset(
                self._data_dir, self._cache_dir, num_workers=self._num_workers, train=False, samples_ids=self._infer_ids
            )

    def train_dataloader(self) -> DataLoader:
        """
        returns train dataloader with class args
        """
        if self._use_batch_sampler:
            # Create a batch sampler for the dataloader
            batch_sampler = BatchSamplerDefault(
                dataset=self._train_dataset,
                balanced_class_name="data.label",
                num_balanced_classes=8,
                batch_size=self._batch_size,
                workers=self._num_workers,
                verbose=True,
            )
            batch_size = 1  # should not provide batch_size for custom batch_sampler (1 is default)

        else:
            batch_sampler = None
            batch_size = self._batch_size

        # Create dataloader
        train_dl = DataLoader(
            dataset=self._train_dataset,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
        )

        return train_dl

    def val_dataloader(self) -> DataLoader:
        """
        returns validation dataloader with class args
        """
        # Create dataloader
        validation_dl = DataLoader(
            dataset=self._validation_dataset,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )

        return validation_dl

    def predict_dataloader(self) -> DataLoader:
        """
        returns validation dataloader with class args
        """
        # Create dataloader
        predict_dl = DataLoader(
            dataset=self._predict_dataset,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return predict_dl


class OpEncodeMetaData(OpBase):
    """
    Encode ISIC 2019 meta-data
    """

    def __init__(self, items_to_encode: List[str] = ["site", "sex", "age"]):
        """
        :param  items_to_encode: which items will be encoded. by default takes all the options.
        """
        super().__init__()

        self._items_to_encode = items_to_encode

    def __call__(self, sample_dict: NDict, key_site: str, key_sex: str, key_age: str, out_prefix: str) -> NDict:
        """
        :param key_site: sample_dict's key for patient's anatom site data
        :param key_sex: sample_dict's key for patient's sex data
        :param key_age: sample_dict's key for patient's age data
        :param out_prefix: the encoded data will be located in sample_dict[f"{out_prefix}.{data_type}"]
        """

        # Encode anatom site into a one-hot vector of length 9
        # 8 anatom sites and 1 for N/A
        if "site" in self._items_to_encode:
            site = sample_dict[key_site]
            site_one_hot = np.zeros(len(ISIC.ANATOM_SITE_INDEX))
            if site in ISIC.ANATOM_SITE_INDEX:
                site_one_hot[ISIC.ANATOM_SITE_INDEX[site]] = 1

            sample_dict[f"{out_prefix}.site"] = site_one_hot

        # Encode sex into a one-hot vector of length 3
        # male, female, N/A
        if "sex" in self._items_to_encode:
            sex = sample_dict[key_sex]
            sex_one_hot = np.zeros(len(ISIC.SEX_INDEX))
            if sex in ISIC.SEX_INDEX:
                sex_one_hot[ISIC.SEX_INDEX[sex]] = 1

            sample_dict[f"{out_prefix}.sex"] = sex_one_hot

        # Encode age into one-hot vector 'u' with length 7 such that:
        # for i in (0, ..., 5), u[i] == 1 iff age in range (20*i, 20(i+1)),
        # and u[6] == 1 iff age has missing value (0> or 120<)
        # for examples:
        #   age 50 -> [0,0,1,0,0,0,0]
        #   age 90 -> [0,0,0,0,1,0,0]
        #   missing age -> [0,0,0,0,0,0,1]
        if "age" in self._items_to_encode:
            age = int(sample_dict[key_age])
            age_one_hot = np.zeros(7)

            encode_idx = age // 20 if (age > 0 and age < 120) else 6
            age_one_hot[encode_idx] = 1

            sample_dict[f"{out_prefix}.age"] = age_one_hot

        return sample_dict
