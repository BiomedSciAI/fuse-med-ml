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


class OpCMMDSampleIDDecode(OpBase):
    """
    decodes sample id into image and segmentation filename
    """

    def __call__(self, sample_dict: NDict) -> NDict:
        """ """
        sid = get_sample_id(sample_dict)

        img_filename_key = "data.input.img_path"
        sample_dict[img_filename_key] = sid

        return sample_dict


class CMMD:
    """
    # dataset that contains breast mamography biopsy info and metadata from chinese patients
    # Path to the stored dataset location
    # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
    # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
    # put on the folliwing in the main folder  -
    # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx
    # 2. folder named CMMD which is the downloaded data folder
    """

    # bump whenever the static pipeline modified
    CMMD_DATASET_VER = 0

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
                (OpCMMDSampleIDDecode(), dict()),  # will save image and seg path to "data.input.img_path"
                (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="dcm")),
                (OpFlipBrightSideOnLeft2D(), dict(key="data.input.img")),
                (OpFindBiggestNonEmptyBbox2D(), dict(key="data.input.img")),
                (OpToNumpy(), dict(key="data.input.img", dtype=np.float32)),
                (OpNormalizeAgainstSelf(), dict(key="data.input.img")),
                (OpResizeAndPad2D(), dict(key="data.input.img", resize_to=(2200, 1200), padding=(60, 60))),
                (
                    OpReadDataframe(
                        data_source,
                        key_column="file",
                        key_name="data.input.img_path",
                        columns_to_extract=["ID1", "file", "classification", "subtype"],
                        rename_columns=dict(
                            ID1="data.patientID", classification="data.gt.classification", subtype="data.gt.subtype"
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
    def merge_clinical_data_with_dicom_tags(data_dir: str, data_misc_dir: str, target: str) -> str:
        """
        Creates a csv file that contains label for each image ( instead of patient as in dataset given file)
        by reading metadata ( breast side and view ) from the dicom files and merging it with the input csv
        If the csv already exists , it will skip the creation proccess
        :param data_dir                     dataset root path
        :param data_misc_dir                path to save misc files to be used later
        :return: the new csv file path
        :return: sample ids of used images
        """
        input_source = os.path.join(data_dir, "CMMD_clinicaldata_revision.csv")
        combined_file_path = os.path.join(data_misc_dir, "files_combined.csv")
        if os.path.isfile(combined_file_path):
            print("Found ground truth file:", combined_file_path)
            merged_clinical_data = pd.read_csv(combined_file_path)
            all_sample_ids = merged_clinical_data["file"].to_list()
            return merged_clinical_data, all_sample_ids
        print("Did not find exising ground truth file!")
        Path(data_misc_dir).mkdir(parents=True, exist_ok=True)
        clinical_data = pd.read_csv(input_source)
        scans = []
        for patient in os.listdir(os.path.join(data_dir, "CMMD")):
            path = os.path.join(data_dir, "CMMD", patient)
            for dicom_file in glob.glob(os.path.join(path, "**/*.dcm"), recursive=True):
                file = dicom_file[len(data_dir) + 1 :] if dicom_file.startswith(data_dir) else ""
                dcm = pydicom.dcmread(os.path.join(data_dir, file))
                scans.append(
                    {
                        "ID1": patient,
                        "LeftRight": dcm[0x00200062].value,
                        "file": file,
                        "view": dcm[0x00540220].value.pop(0)[0x00080104].value,
                    }
                )
        dicom_tags = pd.DataFrame(scans)
        merged_clinical_data = pd.merge(clinical_data, dicom_tags, how="outer", on=["ID1", "LeftRight"])
        merged_clinical_data = merged_clinical_data[merged_clinical_data[target].notna()]
        merged_clinical_data["classification"] = np.where(merged_clinical_data["classification"] == "Benign", 0, 1)
        merged_clinical_data["subtype"] = merged_clinical_data["subtype"].replace(
            {"Luminal A": 0, "Luminal B": 1, "HER2-enriched": 2, "triple negative": 3, np.nan: "4"}
        )
        merged_clinical_data = merged_clinical_data.dropna()
        merged_clinical_data.to_csv(combined_file_path)
        all_sample_ids = merged_clinical_data["file"].to_list()
        return merged_clinical_data, all_sample_ids

    @staticmethod
    def dataset(
        data_dir: str,
        data_misc_dir: str,
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
        :param data_misc_dir                path to save misc files to be used later
        :param target                       target name used from the ground truth dataframe
        :param cache_dir:                   Optional, name of the cache folder
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        :param train: True if used for training  - adds augmentation operations to the pipeline
        :return: DatasetDefault object
        """

        input_source_gt, all_sample_ids = CMMD.merge_clinical_data_with_dicom_tags(data_dir, data_misc_dir, target)

        if sample_ids is None:
            sample_ids = all_sample_ids

        static_pipeline = CMMD.static_pipeline(data_dir, input_source_gt, target)
        dynamic_pipeline = CMMD.dynamic_pipeline(train=train)

        cacher = SamplesCacher(
            "cmmd_cache_ver",
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
