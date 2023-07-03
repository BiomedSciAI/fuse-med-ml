from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.color import OpNormalizeAgainstSelf

from fuseimg.data.ops.aug.geometry import (
    OpAugAffine2D,
    OpAugSqueeze3Dto2D,
    OpAugUnsqueeze3DFrom2D,
)
from fuse.data import PipelineDefault, OpToTensor, OpRepeat
from fuse.data.ops.ops_common import OpLambda, OpLookup, OpToOneHot

from fuse.data.ops.ops_aug_common import OpRandApply, OpSampleAndRepeat
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
from functools import partial
import nibabel as nib
from typing import Hashable, Optional, Sequence
import torch
import pandas as pd
import numpy as np
import skimage
import os
from fuse.data.utils.sample import get_sample_id
from medpy.io import load

from fuse.utils.rand.param_sampler import Uniform, RandInt


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


class OpLoadPICAIImage(OpBase):
    """
    Loads a medical image
    """

    def __init__(
        self, dir_path: str, seuqences: Sequence[str] = ["_t2w"], **kwargs: dict
    ):
        super().__init__(**kwargs)
        self._dir_path = dir_path
        self._sequences = seuqences

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out: str,
    ) -> None:
        """
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out: the key name in sample_dict that holds the image
        :param key_metadata_out : the key to hold metadata dictionary
        """
        try:
            for seq in self._sequences:
                img_filename = os.path.join(
                    self._dir_path,
                    sample_dict[key_in].split("_")[0],
                    sample_dict[key_in] + seq + ".mha",
                )

                image_data, image_header = load(img_filename)
                sample_dict[key_out + seq] = image_data
            return sample_dict
        except:
            print(
                "sample id",
                sample_dict[key_in].split("_")[0],
                "has missing sequence",
                seq,
                "file name does not exist",
                img_filename,
            )
            return None


# loads cancer segmentation - not in use for now
class OpLoadPICAISegmentation(OpBase):
    """
    Loads a medical image
    """

    def __init__(self, data_dir: str, dir_path: str, **kwargs: dict):
        super().__init__(**kwargs)
        self._dir_path = dir_path
        self._data_dir = data_dir

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out: str,
        gt: str,
    ) -> None:
        """
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out: the key name in sample_dict that holds the image
        :param key_metadata_out : the key to hold metadata dictionary
        """
        # for seq in self._sequences:
        seg_filename = os.path.join(self._dir_path, sample_dict[key_in] + ".nii.gz")
        if os.path.exists(seg_filename):
            my_img = nib.load(seg_filename)
            nii_data = my_img.get_fdata()
        else:
            img_filename = os.path.join(
                self._data_dir,
                sample_dict[key_in].split("_")[0],
                sample_dict[key_in] + "_t2w.mha",
            )
            image_data, image_header = load(img_filename)
            nii_data = np.zeros(image_data.shape)
        unique_labels = np.unique(nii_data)
        if sample_dict[gt] not in unique_labels:
            print(
                sample_dict[key_in],
                "ISUP is",
                sample_dict[gt],
                "but unique labels are",
                unique_labels,
            )
            return None
        # TODO - generate different map according to different ISUP in cancer segmentation?
        # else:
        #     output = np.zeros([5]+image_data.shape)
        #     for label in unique_labels:
        #         if label != 0:
        #             output[label-1] = nii_data[nii_data == label]
        sample_dict[key_out] = nii_data
        return sample_dict


class OpLoadPICAISegmentationWholeGland(OpBase):
    """
    Loads a medical image
    """

    def __init__(self, data_dir: str, dir_path: str, **kwargs: dict):
        super().__init__(**kwargs)
        self._dir_path = dir_path
        self._data_dir = data_dir

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str,
        key_out: str,
        gt: str,
    ) -> None:
        """
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out: the key name in sample_dict that holds the image
        :param key_metadata_out : the key to hold metadata dictionary
        """
        # for seq in self._sequences:
        seg_filename = os.path.join(self._dir_path, sample_dict[key_in] + ".nii.gz")
        if os.path.exists(seg_filename):
            my_img = nib.load(seg_filename)
            nii_data = my_img.get_fdata()
        else:
            print(seg_filename, "missing segmentation file")
            return None
        sample_dict[key_out] = nii_data
        return sample_dict


class PICAI:
    """ """

    @staticmethod
    def static_pipeline(
        data_source: pd.DataFrame,
        data_dir: str,
        seg_dir: str,
        target: str,
        repeat_images: Sequence[NDict],
    ) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        repeat_images_with_seg = repeat_images + [(dict(key="data.gt.seg"))]
        bool_map = {"NO": 0, "YES": 1}
        static_pipeline = PipelineDefault(
            "cmmd_static",
            [
                # decoding sample ID
                (
                    OpPICAISampleIDDecode(),
                    dict(),
                ),  # will save image and seg path to "data.input.img_path"
                (
                    OpLoadPICAIImage(data_dir),
                    dict(key_in="data.input.img_path", key_out="data.input.img"),
                ),
                (
                    OpReadDataframe(
                        data_source,
                        key_column="index",
                        key_name="data.input.img_path",
                        #'psa','psad','prostate_volume','histopath_type','lesion_GS','lesion_ISUP','case_ISUP'
                        columns_to_extract=[
                            "index",
                            "patient_id",
                            "study_id",
                            "mri_date",
                            "patient_age",
                            "case_ISUP",
                            "case_csPCa",
                        ],
                        rename_columns=dict(
                            patient_id="data.patientID",
                            case_csPCa="data.gt.classification",
                            case_ISUP="data.gt.subtype",
                        ),
                    ),
                    dict(),
                ),
                (
                    OpLookup(bool_map),
                    dict(
                        key_in="data.gt.classification",
                        key_out="data.gt.classification",
                    ),
                ),
                (
                    OpToOneHot(len(bool_map)),
                    dict(
                        key_in="data.gt.classification",
                        key_out="data.gt.classification_one_hot",
                    ),
                ),
                (
                    OpLoadPICAISegmentationWholeGland(data_dir, seg_dir),
                    dict(
                        key_in="data.input.img_path",
                        key_out="data.gt.seg",
                        gt="data.gt.subtype",
                    ),
                ),
                (
                    OpRepeat(
                        OpLambda(partial(np.transpose, axes=[2, 0, 1])),
                        kwargs_per_step_to_add=repeat_images_with_seg,
                    ),
                    dict(),
                ),
                (
                    OpRepeat(
                        (
                            OpLambda(
                                partial(
                                    skimage.transform.resize,
                                    output_shape=(32, 256, 256),
                                    mode="reflect",
                                    anti_aliasing=True,
                                    preserve_range=True,
                                )
                            )
                        ),
                        kwargs_per_step_to_add=repeat_images_with_seg,
                    ),
                    {},
                ),
                # (OpNormalizeAgainstSelf(), dict(key="data.input.img_t2w")),
                (
                    OpRepeat(
                        (OpNormalizeAgainstSelf()), kwargs_per_step_to_add=repeat_images
                    ),
                    dict(),
                ),
                (
                    OpRepeat((OpToNumpy()), kwargs_per_step_to_add=repeat_images),
                    dict(dtype=np.float32),
                ),
                # (OpResizeAndPad2D(), dict(key="data.input.img", resize_to=(2200, 1200), padding=(60, 60))),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        repeat_images: Sequence[NDict],
        train: bool = False,
        aug_params: NDict = None,
    ) -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train : True iff we request dataset for train purpouse
        """
        repeat_images_with_seg = repeat_images + [dict(key="data.gt.seg")]
        ops = []
        ops += [
            (
                OpRepeat((OpToTensor()), kwargs_per_step_to_add=repeat_images),
                dict(dtype=torch.float32),
            ),
            (OpToTensor(), dict(key="data.gt.seg", dtype=torch.int32)),
            (
                OpRepeat(
                    (OpLambda(partial(torch.unsqueeze, dim=0))),
                    kwargs_per_step_to_add=repeat_images_with_seg,
                ),
                {},
            ),
        ]
        if train:
            ops += [
                # affine augmentation - will apply the same affine transformation on each slice
                (
                    OpRepeat(
                        (OpAugSqueeze3Dto2D()),
                        kwargs_per_step_to_add=repeat_images_with_seg,
                    ),
                    dict(axis_squeeze=1),
                ),
                (
                    OpRandApply(
                        OpSampleAndRepeat(
                            OpAugAffine2D(),
                            kwargs_per_step_to_add=repeat_images_with_seg,
                        ),
                        aug_params["apply_aug_prob"],
                    ),
                    dict(
                        rotate=Uniform(*aug_params["rotate"]),
                        scale=Uniform(*aug_params["scale"]),
                        flip=(aug_params["flip"], aug_params["flip"]),
                        translate=(
                            RandInt(*aug_params["translate"]),
                            RandInt(*aug_params["translate"]),
                        ),
                    ),
                ),
                (
                    OpRepeat(
                        OpAugUnsqueeze3DFrom2D(),
                        kwargs_per_step_to_add=repeat_images_with_seg,
                    ),
                    dict(axis_squeeze=1, channels=1),
                ),
            ]
        dynamic_pipeline = PipelineDefault("picai_dynamic", ops)
        return dynamic_pipeline

    @staticmethod
    def dataset(
        paths: NDict,
        cfg: NDict,
        reset_cache: bool = True,
        sample_ids: Optional[Sequence[Hashable]] = None,
        train: bool = False,
        run_sample: int = 0,
    ) -> DatasetDefault:
        """
        Creates Fuse Dataset single object (either for training, validation and test or user defined set)
        :param paths                        paths dictionary for dataset files
        :param cfg                          dict cfg for training phase
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param sample_ids:                  dataset including the specified sample_ids or None for all the samples.
        :param train:                       True if used for training  - adds augmentation operations to the pipeline
        :param run_sample:                  if > 0 it samples from all the samples #run_sample examples ( used for testing), if =0 then it takes all samples
        :return: DatasetDefault object
        """

        input_source_gt = pd.read_csv(paths["clinical_file"])
        input_source_gt["index"] = (
            input_source_gt["patient_id"].astype(str)
            + "_"
            + input_source_gt["study_id"].astype(str)
        )
        all_sample_ids = input_source_gt["index"].to_list()

        if sample_ids is None:
            sample_ids = all_sample_ids
        if run_sample > 0:
            sample_ids = sample_ids[:run_sample]
        repeat_images = [
            dict(key="data.input.img" + seq) for seq in cfg["series_config"]
        ]
        static_pipeline = PICAI.static_pipeline(
            input_source_gt,
            paths["data_dir"],
            paths["seg_dir"],
            cfg["target"],
            repeat_images,
        )
        if train:
            dynamic_pipeline = PICAI.dynamic_pipeline(
                train=train, repeat_images=repeat_images, aug_params=cfg["aug_params"]
            )
        else:
            dynamic_pipeline = PICAI.dynamic_pipeline(
                train=train, repeat_images=repeat_images
            )

        cacher = SamplesCacher(
            "cache_ver",
            static_pipeline,
            cache_dirs=[paths["cache_dir"]],
            restart_cache=reset_cache,
            audit_first_sample=False,
            audit_rate=None,
            workers=cfg["num_workers"],
        )

        my_dataset = DatasetDefault(
            sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,
        )

        my_dataset.create()
        return my_dataset
