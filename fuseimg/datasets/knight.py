import os
from glob import glob
from typing import Hashable, Optional, Sequence, Tuple

from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
from fuse.utils.ndict import NDict

from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_common import OpLambda, OpZScoreNorm
from fuseimg.data.ops.aug.geometry import OpAugAffine2D, OpRotation3D, OpResizeTo
from fuseimg.data.ops.aug.color import OpAugGaussian
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpClip
import numpy as np
from fuse.data.utils.sample import get_sample_id
from functools import partial
import torch
import pandas as pd


class OpKnightSampleIDDecode(OpBase):
    """
    decodes sample id into image and segmentation filename
    """

    def __call__(self, sample_dict: NDict, test: bool = False) -> NDict:
        """ """

        sid = get_sample_id(sample_dict)
        sample_dict["data.input.case_id"] = sid
        img_filename_key = "data.input.img_path"
        if test:
            sample_dict[img_filename_key] = f"images/{sid}.nii.gz"
        else:
            sample_dict[img_filename_key] = os.path.join(sid, "imaging.nii.gz")

            seg_filename_key = "data.gt.seg_path"
            sample_dict[seg_filename_key] = os.path.join(sid, "aggregated_MAJ_seg.nii.gz")

        return sample_dict


class OpClinicalLoad(OpBase):
    def __init__(self, json_path: str):
        super().__init__()
        self.json_path = json_path

    def __call__(self, sample_dict: NDict, test: bool = False) -> NDict:
        cols = [
            "case_id",
            "age_at_nephrectomy",
            "body_mass_index",
            "gender",
            "comorbidities",
            "smoking_history",
            "radiographic_size",
            "last_preop_egfr",
            "voxel_spacing",
        ]

        if test:
            json_data = pd.read_json(os.path.join(self.json_path, "features.json"))[cols]
        else:
            cols += ["aua_risk_group"]
            json_data = pd.read_json(os.path.join(self.json_path, "knight.json"))[cols]

        sid = sample_dict["data.input.case_id"]
        row = json_data[json_data["case_id"] == sid].to_dict("records")[0]

        row["gender"] = int(row["gender"].lower() == "female")  # female:1 | male:0
        row["comorbidities"] = int(
            any(x for x in row["comorbidities"].values())
        )  # if has any comorbidity it is set to 1
        row["smoking_history"] = ["never_smoked", "previous_smoker", "current_smoker"].index(row["smoking_history"])
        if row["last_preop_egfr"] is None or row["last_preop_egfr"]["value"] is None:
            row["last_preop_egfr"] = 77  # median value
        elif row["last_preop_egfr"]["value"] in (">=90", ">90"):
            row["last_preop_egfr"] = 90
        else:
            row["last_preop_egfr"] = row["last_preop_egfr"]["value"]

        if row["radiographic_size"] is None:
            row["radiographic_size"] = 4.1  # this is the median value on the training set
        if not test:
            sample_dict["data.gt.gt_global.task_1_label"] = int(
                row["aua_risk_group"] in ["high_risk", "very_high_risk"]
            )
            sample_dict["data.gt.gt_global.task_2_label"] = [
                "benign",
                "low_risk",
                "intermediate_risk",
                "high_risk",
                "very_high_risk",
            ].index(row["aua_risk_group"])
        row["voxel_spacing"] = (
            row["voxel_spacing"]["z_spacing"],
            row["voxel_spacing"]["y_spacing"],
            row["voxel_spacing"]["x_spacing"],
        )

        sample_dict["data.input.clinical"] = row
        return sample_dict


class OpPrepareClinical(OpBase):
    def __call__(
        self, sample_dict: NDict
    ) -> NDict:  # , op_id: Optional[str]) -> NDict:, op_id: Optional[str]) -> NDict:
        age = sample_dict["data.input.clinical.age_at_nephrectomy"]
        if age is not None and age > 0 and age < 120:
            age = np.array(age / 120.0).reshape(-1)
        else:
            age = np.array(-1.0).reshape(-1)

        bmi = sample_dict["data.input.clinical.body_mass_index"]
        if bmi is not None and bmi > 10 and bmi < 100:
            bmi = np.array(bmi / 50.0).reshape(-1)
        else:
            bmi = np.array(-1.0).reshape(-1)

        radiographic_size = sample_dict["data.input.clinical.radiographic_size"]
        if radiographic_size is not None and radiographic_size > 0 and radiographic_size < 50:
            radiographic_size = np.array(radiographic_size / 15.0).reshape(-1)
        else:
            radiographic_size = np.array(-1.0).reshape(-1)

        preop_egfr = sample_dict["data.input.clinical.last_preop_egfr"]
        if preop_egfr is not None and preop_egfr > 0 and preop_egfr < 200:
            preop_egfr = np.array(preop_egfr / 90.0).reshape(-1)
        else:
            preop_egfr = np.array(-1.0).reshape(-1)
        # turn categorical features into one hot vectors
        gender = sample_dict["data.input.clinical.gender"]
        gender_one_hot = np.zeros(len(GENDER_INDEX))
        if gender in GENDER_INDEX.values():
            gender_one_hot[gender] = 1

        comorbidities = sample_dict["data.input.clinical.comorbidities"]
        comorbidities_one_hot = np.zeros(len(COMORBIDITIES_INDEX))
        if comorbidities in COMORBIDITIES_INDEX.values():
            comorbidities_one_hot[comorbidities] = 1

        smoking_history = sample_dict["data.input.clinical.smoking_history"]
        smoking_history_one_hot = np.zeros(len(SMOKE_HISTORY_INDEX))
        if smoking_history in SMOKE_HISTORY_INDEX.values():
            smoking_history_one_hot[smoking_history] = 1

        clinical_encoding = np.concatenate(
            (age, bmi, radiographic_size, preop_egfr, gender_one_hot, comorbidities_one_hot, smoking_history_one_hot),
            axis=0,
            dtype=np.float32,
        )
        sample_dict["data.input.clinical.all"] = clinical_encoding
        return sample_dict


class KNIGHT:
    """
    Dataset created for KNIGHT challenge - https://research.ibm.com/haifa/Workshops/KNIGHT/challenge.html
    Aims to predict the risk level of patients based on CT scan and/or clinical data.
    """

    @staticmethod
    def sample_ids(path: str) -> list:
        """
        get all the sample ids in train-set
        sample_id is directory file named case_xxxxx found in the specified path
        """
        files = [os.path.basename(f) for f in glob(os.path.join(path, "case_*"))]
        return files

    @staticmethod
    def static_pipeline(data_path: str, resize_to: Tuple, test: bool = False) -> PipelineDefault:
        static_pipeline = PipelineDefault(
            "static",
            [
                # decoding sample ID
                (
                    OpKnightSampleIDDecode(),
                    dict(test=test),
                ),  # will save image and seg path to "data.input.img_path", "data.gt.seg_path" and load json data
                (OpClinicalLoad(data_path), dict(test=test)),
                # loading data
                (OpLoadImage(data_path), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
                # fixed image normalization
                (OpClip(), dict(key="data.input.img", clip=(-62, 301))),
                (OpZScoreNorm(), dict(key="data.input.img", mean=104.0, std=75.3)),  # kits normalization
                # transposing so the depth channel will be first
                (
                    OpLambda(partial(np.moveaxis, source=-1, destination=0)),
                    dict(key="data.input.img"),
                ),  # convert image from shape [H, W, D] to shape [D, H, W]
                (OpPrepareClinical(), dict()),  # process clinical data
                (OpResizeTo(channels_first=False), dict(key="data.input.img", output_shape=resize_to)),
            ],
        )
        return static_pipeline

    @staticmethod
    def train_dynamic_pipeline() -> PipelineDefault:
        train_dynamic_pipeline = PipelineDefault(
            "dynamic",
            [
                # Numpy to tensor
                (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
                (OpToTensor(), dict(key="data.input.clinical.all")),
                (
                    OpRandApply(OpSample(OpRotation3D()), 0.5),
                    dict(
                        key="data.input.img",
                        z_rot=Uniform(-5.0, 5.0),
                        x_rot=Uniform(-5.0, 5.0),
                        y_rot=Uniform(-5.0, 5.0),
                    ),
                ),
                # affine transformation per slice but with the same arguments
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
                (OpAugGaussian(), dict(key="data.input.img", std=0.01)),
                # add channel dimension -> [C=1, D, H, W]
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
            ],
        )
        return train_dynamic_pipeline

    @staticmethod
    def val_dynamic_pipeline() -> PipelineDefault:
        val_dynamic_pipeline = PipelineDefault(
            "dynamic",
            [
                # Numpy to tensor
                (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
                (OpToTensor(), dict(key="data.input.clinical.all")),
                # add channel dimension -> [C=1, D, H, W]
                (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")),
            ],
        )
        return val_dynamic_pipeline

    @staticmethod
    def dataset(
        data_path: str = "data",
        cache_dir: str = "cache",
        split: dict = None,
        sample_ids: Optional[Sequence[Hashable]] = None,
        test: bool = False,
        reset_cache: bool = False,
        resize_to: Tuple = (70, 256, 256),
    ) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param split: dictionary including sample ids for (train and validation) or test.
        :param sample_ids: dataset including the specified sample_ids. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        either split or sample_ids is not None. there is no need in both of them.
        :param test: boolean indicating weather to use train dynamic pipeline or val. only necessary when using sample_ids param.
        :param reset_cache: set to True tp reset the cache
        :param train: True if used for training  - adds augmentation operations to the pipeline
        """
        train_dynamic_pipeline = KNIGHT.train_dynamic_pipeline()
        val_dynamic_pipeline = KNIGHT.val_dynamic_pipeline()

        # Create dataset
        if sample_ids is not None:
            static_pipeline = KNIGHT.static_pipeline(data_path, resize_to=resize_to, test=test)
            cacher = SamplesCacher(
                "cache", static_pipeline, cache_dirs=[f"{cache_dir}/data"], restart_cache=reset_cache, workers=8
            )
            dataset = DatasetDefault(
                sample_ids=sample_ids,
                static_pipeline=static_pipeline,
                dynamic_pipeline=val_dynamic_pipeline if test else train_dynamic_pipeline,
                cacher=cacher,
            )
            print("- Load and cache data:")
            dataset.create()
            print("- Load and cache data: Done")
            return dataset

        static_pipeline = KNIGHT.static_pipeline(data_path, resize_to=resize_to, test=("test" in split))
        if "train" in split:
            train_cacher = SamplesCacher(
                "train_cache", static_pipeline, cache_dirs=[f"{cache_dir}/train"], restart_cache=reset_cache, workers=8
            )

            train_dataset = DatasetDefault(
                sample_ids=split["train"],
                static_pipeline=static_pipeline,
                dynamic_pipeline=train_dynamic_pipeline,
                cacher=train_cacher,
            )

            print("- Load and cache data:")
            train_dataset.create()

            print("- Load and cache data: Done")

            print("Train Data: Done", {"attrs": "bold"})

            #### Validation data
            print("Validation Data:", {"attrs": "bold"})

            val_cacher = SamplesCacher(
                "val_cache", static_pipeline, cache_dirs=[f"{cache_dir}/val"], restart_cache=reset_cache, workers=8
            )
            ## Create dataset
            validation_dataset = DatasetDefault(
                sample_ids=split["val"],
                static_pipeline=static_pipeline,
                dynamic_pipeline=val_dynamic_pipeline,
                cacher=val_cacher,
            )

            print("- Load and cache data:")
            validation_dataset.create()
            print("- Load and cache data: Done")

            print("Validation Data: Done", {"attrs": "bold"})

            return train_dataset, validation_dataset
        else:  # test only
            #### Test data
            print("Test Data:", {"attrs": "bold"})

            ## Create dataset
            test_dataset = DatasetDefault(
                sample_ids=split["test"],
                static_pipeline=static_pipeline,
                dynamic_pipeline=val_dynamic_pipeline,
            )

            print("- Load and cache data:")
            test_dataset.create()
            print("- Load and cache data: Done")

            print("Test Data: Done", {"attrs": "bold"})
            return test_dataset


GENDER_INDEX = {"male": 0, "female": 1}
COMORBIDITIES_INDEX = {"no comorbidities": 0, "comorbidities exist": 1}
SMOKE_HISTORY_INDEX = {"never smoked": 0, "previous smoker": 1, "current smoker": 2}
