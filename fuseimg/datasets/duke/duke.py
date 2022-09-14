import numpy as np
import glob
import os
from typing import Any, Callable, Hashable, Optional, Sequence, List, Dict

# import fuseimg.data.ops.ops_common_imaging
from fuse.data.ops.ops_common import OpLambda, OpKeepKeypaths
from functools import partial

from fuseimg.data.ops.color import OpNormalizeAgainstSelf, OpClip, OpToRange
from fuseimg.data.ops.aug.geometry_3d import OpRotation3D
from fuseimg.data.ops.aug.geometry import OpAugSqueeze3Dto2D, OpAugAffine2D, OpAugUnsqueeze3DFrom2D
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform

import pandas as pd

from fuse.data import DatasetDefault
from fuse.data import PipelineDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.utils.sample import get_sample_id, get_sample_id_key
from fuse.utils import NDict
from fuseimg.data.ops.ops_mri import (
    OpExtractDicomsPerSeq,
    OpLoadDicomAsStkVol,
    OpExtractRadiomics,
    OpSelectVolumes,
    OpResampleStkVolsBasedRef,
    OpStackList4DStk,
    OpRescale4DStk,
    OpExtractPatchAnotations,
    OpAddMaskFromBoundingBoxAsLastChannel,
    OpCreatePatchVolumes,
    OpStk2Dict,
    OpStk2DictSingle,
    OpDict2Stk,
    OpDict2StkSingle,
    OpSortSequence,
    OpGroupSequences,
    OpDeleteSequenceAttr,
    OpLoadDicomsAsStkVolByPath,
    extract_series_num,
    extract_dicom_field,
    sort_dicoms_by_field,
)
from fuseimg.data.ops.ops_debug import OpVis2DImage, OpVisImageHist

import torch
import pydicom

from fuseimg.datasets.duke.duke_label_type import DukeLabelType

from fuse.data.ops.ops_debug import OpPrintTypes, OpPrintKeysContent


def get_selected_series_index(sample_id: List[str], seq_id: str) -> List[int]:  # Not so sure about types
    """
    TODO

    :param sample_id:
    :param seq_id:
    """
    patient_id = sample_id[0]
    if patient_id in ["Breast_MRI_120", "Breast_MRI_596"]:
        map = {"DCE_mix": [2], "MASK": [0]}  # NO MASK in seq ids
        # idx = 2
    else:
        map = {"DCE_mix": [1], "MASK": [0]}  # NO MASK in seq ids
        # idx = 1
    return map[seq_id]


class Duke:
    DUKE_DATASET_VER = 0

    @staticmethod
    def dataset(
        label_type: Optional[DukeLabelType] = None,
        train: Optional[int] = False,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        select_series_func: Callable = get_selected_series_index,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        add_clinical_features: Optional[bool] = False,
        verbose: Optional[bool] = False,
        cache_kwargs: Optional[dict] = None,
    ) -> DatasetDefault:

        """
        :param label_type: type of label to use
        :param train: set 'True' to apply augmentations
        :param cache_dir: path to store the cache of the static pipeline
        :param data_dir: path to the original data
        :param select_series_func: which series to select for DCE_mix sequences
        :param reset_cache:
        :param num_workers:  number of processes used for caching
        :param sample_ids: list of selected patient_ids for the dataset
        :param add_clinical_features:
        :param verbose: TODO
        :param cache_kwargs: TODO
        :return:
        """

        if sample_ids is None:
            sample_ids = Duke.sample_ids()

        static_pipeline = Duke.static_pipeline(
            data_dir=data_dir, select_series_func=select_series_func, verbose=verbose
        )
        dynamic_pipeline = Duke.dynamic_pipeline(
            data_dir=data_dir,
            train=train,
            label_type=label_type,
            verbose=verbose,
            add_clinical_features=add_clinical_features,
        )

        if cache_dir is None:
            cacher = None
        else:
            if cache_kwargs is None:
                cache_kwargs = {}
            cacher = SamplesCacher(
                f"duke_cache_ver{Duke.DUKE_DATASET_VER}",
                static_pipeline,
                [cache_dir],
                restart_cache=reset_cache,
                workers=num_workers,
                ignore_nan_inequality=True,
                **cache_kwargs,
            )

        my_dataset = DatasetDefault(
            sample_ids=sample_ids, static_pipeline=static_pipeline, dynamic_pipeline=dynamic_pipeline, cacher=cacher
        )
        my_dataset.create()
        return my_dataset

    @staticmethod
    def sample_ids() -> List[str]:
        """
        Returns all the 922 sample IDs for the Duke Breast MRI dataset
        """
        return [f"Breast_MRI_{i:03d}" for i in range(1, 923)]

    @staticmethod
    def static_pipeline(
        select_series_func: Callable,
        data_dir: Optional[str] = None,
        with_rescale: Optional[bool] = True,
        output_stk_volumes: Optional[bool] = False,
        output_patch_volumes: Optional[bool] = True,
        verbose: Optional[bool] = True,
        duke_patch_annotations_df: Optional[pd.DataFrame] = None,
        name_suffix: str = "",
    ) -> PipelineDefault:
        """
        :param select_series_func:
        :param data_dir:
        :param with_rescale:
        :param output_stk_volumes:
        :param output_patch_volumes:
        :param verbose:
        :param duke_patch_annotations_df:
        :param name_suffix:
        """
        data_dir = os.environ["DUKE_DATA_PATH"] if data_dir is None else data_dir
        mri_dir = os.path.join(data_dir, "manifest-1607053360376")
        mri_dir2 = os.path.join(mri_dir, "Duke-Breast-Cancer-MRI")
        metadata_path = os.path.join(mri_dir, "metadata.csv")

        series_desc_2_sequence_map = get_series_desc_2_sequence_mapping(metadata_path)
        seq_ids = ["DCE_mix_ph1", "DCE_mix_ph2", "DCE_mix_ph3", "DCE_mix_ph4", "DCE_mix", "DCE_mix_ph"]

        # Function for sorting 'OpSortSequence'
        def sort_DCE_mix_ph_func(seq_info_list: Dict[str, Any]) -> Any:
            return seq_info_list["series_num"]

        def image_process_func(img: np.ndarray):
            return img[:,:,79]

        static_pipeline_steps = [
            # step 1: map sample_id to paths of MRI image
            (
                OpDukeSampleIDDecode(data_path=mri_dir2, series_desc_2_sequence_map=series_desc_2_sequence_map),
                dict(key_out="data.input.mri_path", sort_dicoms=False),
            ),
            # step 2: read dicom paths (sorting)
            # (
            #     OpExtractDicomsPerSeq(
            #         seq_ids=seq_ids, series_desc_2_sequence_map=series_desc_2_sequence_map, use_order_indicator=False
            #     ),
            #     dict(
            #         key_in="data.input.mri_path",
            #         key_out_sequence_prefix="data.input.sequence",
            #     ),
            # ),
            # step 3: Load STK volumes of MRI sequences
            # (
            #     OpLoadDicomAsStkVol(seq_ids=seq_ids),
            #     dict(key_sequence_prefix="data.input.sequence", key_volume="stk_volume"),
            # ),
            # step 4: group DCE sequences into DCE_mix
            # (
            #     OpSortSequence(key_sort=sort_DCE_mix_ph_func),
            #     dict(key_in_seq_prefix="data.input.sequence", key_in_seq_id="DCE_mix_ph"),
            # ),
            # (
            #     OpGroupSequences(delete_source_seq=True),
            #     dict(
            #         ids_source=["DCE_mix_ph1", "DCE_mix_ph2", "DCE_mix_ph3", "DCE_mix_ph4", "DCE_mix_ph"],
            #         id_target="DCE_mix",
            #         key_sequence_prefix="data.input.sequence",
            #     ),
            # ),
            # step 5: select single volume from DCE_mix sequence
            # (
            #     OpSelectVolumes(
            #         get_indexes_func=select_series_func,
            #         selected_seq_ids=["DCE_mix"],
            #     ),
            #     dict(
            #         key_in_sequence_prefix="data.input.sequence",
            #         key_in_volume="stk_volume",
            #         key_out_volumes="data.input.selected_volumes",
            #         key_out_volumes_info="data.input.selected_volumes_info",
            #     ),
            # ),
            # (
            #     OpDeleteSequenceAttr(seq_ids=seq_ids),
            #     dict(key_in_seq_prefix="data.input.sequence", attribute="stk_volume"),
            # ),
            # step 6: set first volume to be the reference volume and register other volumes with respect to it
            # (
            #     OpResampleStkVolsBasedRef(reference_inx=0, interpolation="bspline"),
            #     dict(key="data.input.selected_volumes"),
            # ),
            # step 7: create a single 4D volume from all the sequences (4th channel is the sequences channel)
            # (
            #     OpStackList4DStk(delete_input_volumes=True),
            #     dict(
            #         key_in="data.input.selected_volumes",
            #         key_out_volume4d="data.input.volume4D",
            #         key_out_ref_volume="data.input.ref_volume",
            #     ),
            # ),
            (OpLoadDicomsAsStkVolByPath(), dict(key_in="data.input.sub_path", key_out="data.input.stk_volume")),
            # (OpPrintKeysContent(num_samples=1), dict(keys=None)),
            (OpNormalizeAgainstSelf(), dict(key="data.input.stk_volume")),
            # (OpVisImageHist(num_samples=1, show=False), dict(key="data.input.stk_volume", bins=10)),
            # (OpVis2DImage(num_samples=1, show=False, image_process_func=image_process_func), dict(key="data.input.stk_volume")),
            (
                OpStackList4DStk(delete_input_volumes=True),
                dict(
                    key_in="data.input.stk_volume",
                    key_out_volume4d="data.input.volume4D",
                    key_out_ref_volume="data.input.ref_volume",
                ),
            ),
        ]

        # if with_rescale:
        #     # step 8: rescale the 4d volume
        #     static_pipeline_steps += [(OpRescale4DStk(), dict(key="data.input.volume4D"))]


        # step 9: read raw annotations - will be used for labels, features, and also for creating lesion properties
        annotations_path = os.path.join(data_dir, "Annotation_Boxes.csv")
        static_pipeline_steps += [
            (
                # OpReadDataframe(data=get_duke_raw_annotations_df(data_dir), key_column="Patient ID"),
                OpReadDataframe(data_filename=annotations_path, key_column="Patient ID"),
                dict(key_out_group="data.input.annotations"),
            )
        ]

        # step 10: add lesion properties for each sample id
        if output_patch_volumes:
            if duke_patch_annotations_df is not None:  # TODO figure if it is relevant or not! seems not to be in use
                # read previousy computed patch annotations
                static_pipeline_steps += [
                    (
                        OpReadDataframe(data=duke_patch_annotations_df, key_column="Patient ID"),
                        dict(key_out_group="data.input.patch_annotations"),
                    ),
                ]
            else:
                # generate patch annotations
                static_pipeline_steps += [
                    # add lesion features
                    (
                        OpExtractPatchAnotations(),
                        dict(
                            key_in_ref_volume="data.input.ref_volume",
                            key_in_annotations="data.input.annotations",
                            key_out="data.input.patch_annotations",
                        ),
                    )
                ]

            static_pipeline_steps += [
                # step 11: generate a mask from the lesion BB and append it as a new (last) channel
                (
                    OpAddMaskFromBoundingBoxAsLastChannel(name_suffix=name_suffix),
                    dict(
                        key_volume4D="data.input.volume4D",
                        key_in_ref_volume="data.input.ref_volume",
                        key_in_patch_annotations="data.input.patch_annotations",
                    ),
                ),
                # step 12: create patch volumes using the mask channel:
                #    (i) fixed size (original scale) around center of annotatins (orig),
                #    and (ii) entire annotations
                (
                    OpCreatePatchVolumes(
                        lesion_shape=(9, 100, 100),
                        lesion_spacing=(1, 0.5, 0.5),
                        crop_based_annotation=True,
                        name_suffix=name_suffix,
                        delete_input_volumes=not output_stk_volumes,
                    ),
                    dict(
                        key_in_volume4D="data.input.volume4D",
                        key_in_ref_volume="data.input.ref_volume",
                        key_in_patch_annotations="data.input.patch_annotations",
                        key_out="data.input.patch_volume",
                    ),
                ),
            ]
        if output_stk_volumes:
            static_pipeline_steps += [
                # step 13: move STK volumes to ndarrays - to allow quick saving to disk
                (OpStk2Dict(), dict(keys=["data.input.volume4D", "data.input.ref_volume"]))
            ]
        static_pipeline = PipelineDefault("static", static_pipeline_steps, verbose=verbose)

        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        data_dir: Optional[str] = None,
        label_type: Optional[DukeLabelType] = None,
        train: Optional[bool] = False,
        num_channels: Optional[int] = 1,
        verbose: Optional[bool] = True,
        add_clinical_features: Optional[bool] = False,  # TODO this one
    ) -> PipelineDefault:
        """
        TODO fill
        :param data_dir:
        :param label_type:
        :param train:
        :param num_channels:
        :param verbose:
        :param add_clinical_features:
        """
        data_dir = os.environ["DUKE_DATA_PATH"] if data_dir is None else data_dir
        volume_key = "data.input.patch_volume"

        def delete_last_channel_in_volume(sample_dict: NDict) -> NDict:
            vol = sample_dict[volume_key]
            vol = vol[:-1]  # remove last channel
            sample_dict[volume_key] = vol
            return sample_dict

        dynamic_steps = [
            # step 1: delete the mask channel
            (OpLambda(func=delete_last_channel_in_volume), dict(key=None)),
            # step 2: turn volume to tensor
            (OpToTensor(), dict(key="data.input.patch_volume", dtype=torch.float32)),
        ]
        if train:
            # step 3: augmentations
            dynamic_steps += [
                # step 3.1. 3D rotation
                (
                    OpRandApply(OpSample(OpRotation3D()), 0.5),
                    dict(
                        key="data.input.patch_volume",
                        ax1_rot=Uniform(-5.0, 5.0),
                        ax2_rot=Uniform(-5.0, 5.0),
                        ax3_rot=Uniform(-5.0, 5.0),
                    ),
                ),
                # step 3.2.1 3D => 2D
                # TODO: revisit this is the right axis (tal used 'z'). same for 3.2.3
                (OpAugSqueeze3Dto2D(), dict(key="data.input.patch_volume", axis_squeeze=1)),
                # step 3.2.2 2D affine transformation
                (
                    OpRandApply(OpSample(OpAugAffine2D()), 0.5),
                    dict(
                        key="data.input.patch_volume",
                        rotate=Uniform(0, 360.0),
                        scale=Uniform(0.9, 1.1),
                        flip=(RandBool(0.5), RandBool(0.5)),
                        translate=(RandInt(-4, 4), RandInt(-4, 4)),
                    ),
                ),
                # step 3.2.3 2D => 3D
                (
                    OpAugUnsqueeze3DFrom2D(),
                    dict(
                        key="data.input.patch_volume",
                        axis_squeeze=1,
                        channels=num_channels,
                    ),
                ),
            ]

        sid_key = get_sample_id_key()  # TODO added for the test dataset
        keys_2_keep = [volume_key, sid_key]
        key_ground_truth = "data.ground_truth"
        if add_clinical_features or (label_type is not None):

            # step 4 (optional): read clinical data
            dynamic_steps += [
                (
                    OpReadDataframe(
                        data=get_duke_clinical_data_df(data_dir), key_column="Patient Information:Patient ID"
                    ),
                    dict(key_out_group="data.input.clinical_data"),
                ),
                # (OpPrintKeysContent(num_samples=1), dict(keys=None)),  # DEBUG
            ]
            if label_type is not None:
                # dynamic_steps.append((OpAddDukeLabelAndClinicalFeatures(label_type=label_type),
                #                 dict(key_in='data.input.clinical_data',  key_out_gt=key_ground_truth,
                #                      key_out_clinical_features='data.clinical_features')))
                #
                # step 5: add ground truth label
                dynamic_steps.append(
                    (
                        OpLambda(func=label_type.get_value),
                        dict(key="data.input.clinical_data", key_out=key_ground_truth),
                    )
                )
                # dynamic_steps.append((OpLambda(func=debug_print_keys), dict(key=None))) # TODO delete when finish PR

                # step 6: remove entries with Nan labels
                dynamic_steps.append(
                    (
                        OpLambda(
                            func=partial(remove_entries_with_nan_label, key=key_ground_truth)
                        ),  # TODO double check with moshiko (return None)
                        dict(key=None),
                    )
                )
                keys_2_keep.append(key_ground_truth)

            if add_clinical_features and label_type is not None:
                key_clinical_features = "data.clinical_features"
                dynamic_steps.append(
                    (
                        OpLambda(func=label_type.select_features),
                        dict(key="data.input.clinical_data", key_out=key_clinical_features),
                    )
                )
                keys_2_keep.append(key_clinical_features)

            for key in keys_2_keep:
                if key == volume_key or key == "data.sample_id":
                    continue
                dtype = torch.int64 if key == key_ground_truth else torch.float32
                dynamic_steps += [(OpToTensor(), dict(key=key, dtype=dtype))]
        dynamic_steps.append((OpKeepKeypaths(), dict(keep_keypaths=keys_2_keep)))

        dynamic_pipeline = PipelineDefault("dynamic", dynamic_steps, verbose=verbose)
        print("Init Dynamic pipeline: Done")
        return dynamic_pipeline


class OpDukeSampleIDDecode(OpBase):
    """
    decodes sample id into path of the MRI image
    """

    def __init__(self, data_path: str, series_desc_2_sequence_map: Dict[str, str]):
        """
        :param data_path:
        :param series_desc_2_sequence_map:
        """
        super().__init__()
        self._data_path = data_path
        self._series_desc_2_sequence_map = series_desc_2_sequence_map

    def __call__(self, sample_dict: NDict, key_out: str, sort_dicoms: bool) -> NDict:
        sid = get_sample_id(sample_dict)
        idx = 2 if sid in ["Breast_MRI_120", "Breast_MRI_596"] else 1

        sample_path = get_sample_path(self._data_path, sid)
        sample_dict[key_out] = sample_path

        # Which seq_ids have dcm files?
        seq_ids_to_paths = dict()
        seq_2_info_dict = dict()
        for seq_dir in os.listdir(sample_path):
            seq_path = os.path.join(sample_path, seq_dir)

            # get dcm file in dir
            dcm_file = glob.glob(os.path.join(seq_path, "*.dcm"))[0]
            dcm_ds = pydicom.dcmread(dcm_file)

            # Read series description from one of the dicoms
            series_desc = dcm_ds.SeriesDescription

            # Derive the sequence ID from the series description
            seq_id = self._series_desc_2_sequence_map.get(series_desc, "UNKNOWN")

            series_num = extract_series_num(dcm_ds)  # TODO maybe take those as input (?)
            dicom_field = extract_dicom_field(dcm_ds, seq_id)


            seq_2_info_dict.setdefault(seq_id, [])
            seq_2_info_dict[seq_id].append(
                dict(path=seq_path, series_num=series_num, dicom_field=dicom_field, series_desc=series_desc)
            )

            seq_ids_to_paths.setdefault(seq_id, [])
            seq_ids_to_paths[seq_id].append(seq_path)

        # iterate by the wanted order and choose the sub-path to the relevant seq_dir
        present_seq_ids = seq_ids_to_paths.keys()
        wanted_sequences = ["DCE_mix_ph1", "DCE_mix_ph2", "DCE_mix_ph3", "DCE_mix_ph4", "DCE_mix_ph"]
        iter_sequences = [seq_id for seq_id in wanted_sequences if seq_id in present_seq_ids]
        if iter_sequences == []:
            print(f"Didn't found sequence data for sample ({sid}).")
            return
            # raise Exception(f"Didn't found sequence data for sample ({sid}).")

        # save the sub-path
        wanted_seq_id = iter_sequences[0]  # OK
        seq_sub_path = seq_ids_to_paths[wanted_seq_id][0]  # NAIVE !

        # not naive:
        if sort_dicoms:
            wanted_seq_info_list = seq_2_info_dict[wanted_seq_id]
            seq_info = wanted_seq_info_list[0]

            dicom_group_ids, sorted_dicom_groups = sort_dicoms_by_field(
                seq_path=seq_info["path"],
                dicom_field=seq_info["dicom_field"],
                use_order_indicator=False
            )
            for dicom_group_id, dicom_group in zip(dicom_group_ids, sorted_dicom_groups):
                seq_info2 = dict(
                    path=seq_info["path"],
                    series_num=seq_info["series_num"],
                    series_desc=seq_info["series_desc"],
                    dicoms=dicom_group,  # each sequence/series path may contain several (sub-)sequence/series
                    dicoms_id=dicom_group_id,
                )

                seq_sub_path = seq_info2["path"]
                break



        sample_dict["data.input.sub_path"] = seq_sub_path
        return sample_dict


class OpAddDukeLabelAndClinicalFeatures(OpBase):
    """
    decodes sample id into path of MRI images
    """

    def __init__(self, label_type: DukeLabelType, is_concat_features_to_input: Optional[bool] = False, **kwargs: Any):
        super().__init__(**kwargs)
        self._label_type = label_type
        self._is_concat_features_to_input = is_concat_features_to_input

    def __call__(self, sample_dict: NDict, key_in: str, key_out_gt: str, key_out_clinical_features: str) -> NDict:
        clinical_features = sample_dict[key_in]
        label_val = self._label_type.get_value(clinical_features)
        if np.isnan(label_val):
            return None  # should filter example (instead of sample_dict['data.filter'] = True )

        label_tensor = torch.tensor(label_val, dtype=torch.int64)
        sample_dict[key_out_gt] = label_tensor  # 'data.ground_truth'

        # add clinical
        features_to_use = self._label_type.select_features()

        clinical_features_to_use = torch.tensor(
            [float(clinical_features[feature]) for feature in features_to_use], dtype=torch.float32
        )
        sample_dict[key_out_clinical_features] = clinical_features_to_use  # 'data.clinical_features'

        if self._is_concat_features_to_input:
            # select input channel
            input_tensor = sample_dict["data.input"]
            input_shape = input_tensor.shape
            for feature in clinical_features_to_use:
                input_tensor = torch.cat(
                    (input_tensor, feature.repeat(input_shape[1], input_shape[2], input_shape[3]).unsqueeze(0)), dim=0
                )
            sample_dict["data.input"] = input_tensor

        return sample_dict


def get_duke_clinical_data_df(duke_data_dir: str) -> pd.DataFrame:
    """
    TODO
    :param duke_data_dir:
    """
    clinical_path = os.path.join(duke_data_dir, "Clinical_and_Other_Features.xlsx")
    df = pd.read_excel(clinical_path, sheet_name="Data", nrows=10)

    columns = []
    col_header = ""
    for i in range(df.shape[1]):
        if not df.columns[i].startswith("Unnamed"):
            col_header = df.columns[i]
        col_name = col_header.strip()

        for row in [0, 1]:
            s = df.iloc[row, i]
            if isinstance(s, str) and len(s.strip()) > 0:
                col_name += ":" + s.strip()
            if col_name not in columns:
                break
        columns.append(col_name)

    clinical_df = pd.read_excel(clinical_path, sheet_name="Data", skiprows=3, header=None, names=columns)
    return clinical_df


# TODO delete (?)
def get_samples_for_debug(
    data_dir: str, n_pos: int, n_neg: int, label_type: DukeLabelType, sample_ids: List[str] = None
) -> List[str]:
    """
    TODO

    :param data_dir:
    :param n_pos:
    :param n_neg:
    :param label_type:
    :param sample_ids:
    """
    annotations_df = get_duke_clinical_data_df(data_dir).set_index("Patient Information:Patient ID")
    if sample_ids is not None:
        annotations_df = annotations_df.loc[sample_ids]
    label_values = label_type.get_value(annotations_df)
    patient_ids = annotations_df.index
    debug_sample_ids = []
    for label_val, n_vals in zip([True, False], [n_pos, n_neg]):
        debug_sample_ids += patient_ids[label_values == label_val].values.tolist()[:n_vals]
    return debug_sample_ids


def get_series_desc_2_sequence_mapping(metadata_path: str) -> Dict[str, str]:
    """
    Read metadata file and match between series_desc in metadata file and sequence

    :param metadata_path: path to metadata
    """

    metadata_df = pd.read_csv(metadata_path)
    series_description_list = metadata_df["Series Description"].unique()

    series_desc_2_sequence_mapping = {"ax dyn": "DCE_mix_ph"}

    patterns = ["1st", "2nd", "3rd", "4th"]
    for i_phase in range(1, 5):
        seq_id = f"DCE_mix_ph{i_phase}"
        phase_patterns = [patterns[i_phase - 1], f"{i_phase}ax", f"{i_phase}Ax", f"{i_phase}/ax", f"{i_phase}/Ax"]

        for series_desc in series_description_list:
            has_match = any(p in series_desc for p in phase_patterns)
            if has_match:
                series_desc2 = series_desc.replace(f"{i_phase}ax", f"{i_phase}/ax").replace(
                    f"{i_phase}Ax", f"{i_phase}/Ax"
                )
                series_desc_2_sequence_mapping[series_desc] = seq_id
                series_desc_2_sequence_mapping[series_desc2] = seq_id

    return series_desc_2_sequence_mapping


def get_sample_path(data_path: str, sample_id: str) -> str:
    """
    Returns sample's path.
    Note that each sample's folder contains another folder which contains the data itself.

    :param data_path: path to the data directory
    :param sample_id: sample's id
    """
    sample_path_pattern = os.path.join(data_path, sample_id, "*")
    sample_path = glob.glob(sample_path_pattern)
    assert len(sample_path) == 1

    return sample_path[0]


##################################
class DukeRadiomics(Duke):
    @staticmethod
    def static_pipeline(
        select_series_func: Callable, data_dir: Optional[str] = None, verbose: Optional[bool] = True
    ) -> PipelineDefault:
        # remove scaling operator for radiomics calculation
        static_pipline = Duke.static_pipeline(
            data_dir=data_dir,
            select_series_func=select_series_func,
            with_rescale=False,
            output_patch_volumes=False,
            output_stk_volumes=True,
            verbose=verbose,
        )
        return static_pipline

    @staticmethod
    def dynamic_pipeline(
        radiomics_extractor_setting: dict, label_type: Optional[DukeLabelType] = None, verbose: Optional[bool] = False
    ) -> PipelineDefault:

        dynamic_steps = [
            (OpDict2Stk(), dict(keys=["data.input.volume4D", "data.input.ref_volume"])),
            (
                OpExtractRadiomics(extractor_settings=radiomics_extractor_setting),
                dict(key_in_vol_4d="data.input.volume4D", key_out_radiomics_results="data.radiomics"),
            ),
        ]

        keys_2_keep = ["data.radiomics.original"]
        if label_type is not None:
            key_ground_truth = "data.ground_truth"
            dynamic_steps.append(
                (
                    OpAddDukeLabelAndClinicalFeatures(label_type=label_type),
                    dict(
                        key_in="data.input.annotations",
                        key_out_gt=key_ground_truth,
                        key_out_clinical_features="data.clinical_features",
                    ),
                )
            )
            keys_2_keep.append(key_ground_truth)
        dynamic_steps.append((OpKeepKeypaths(), dict(keep_keypaths=keys_2_keep)))

        dynamic_pipeline = PipelineDefault("dynamic", dynamic_steps, verbose=verbose)

        return dynamic_pipeline

    @staticmethod
    def dataset(
        radiomics_extractor_setting: dict,
        label_type: Optional[DukeLabelType] = None,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        select_series_func: Callable = get_selected_series_index,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        verbose: Optional[bool] = True,
        cache_kwargs: Optional[dict] = None,
    ) -> DatasetDefault:

        """
        :param label_type: type of label to use
        :param cache_dir: path to store the cache of the static pipeline
        :param data_dir: path to the original data
        :param select_series_func: which series to select for DCE_mix sequences
        :param reset_cache:
        :param num_workers:  number of processes used for caching
        :param sample_ids: list of selected patient_ids for the dataset
        :return:
        """

        if sample_ids is None:
            sample_ids = DukeRadiomics.sample_ids()

        static_pipeline = DukeRadiomics.static_pipeline(
            data_dir=data_dir, select_series_func=select_series_func, verbose=verbose
        )
        dynamic_pipeline = DukeRadiomics.dynamic_pipeline(
            radiomics_extractor_setting=radiomics_extractor_setting, label_type=label_type, verbose=verbose
        )

        if cache_dir is None:
            cacher = None
        else:
            if cache_kwargs is None:
                cache_kwargs = {}
            cacher = SamplesCacher(
                f"duke_cache_ver{Duke.DUKE_DATASET_VER}",
                static_pipeline,
                [cache_dir],
                restart_cache=reset_cache,
                workers=num_workers,
                ignore_nan_inequality=True,
                **cache_kwargs,
            )

        my_dataset = DatasetDefault(
            sample_ids=sample_ids, static_pipeline=static_pipeline, dynamic_pipeline=dynamic_pipeline, cacher=cacher
        )
        my_dataset.create()
        return my_dataset


def remove_entries_with_nan_label(sample_dict: NDict, key: str) -> NDict:
    """
    TODO

    :param sample_dict:
    :param key:
    """
    if np.isnan(sample_dict[key]):
        print(f"====== {get_sample_id(sample_dict)} has nan label ==> excluded")
        return None
    return sample_dict


# TODO delete (?) seems not to be in use
def get_selected_sample_ids() -> List[str]:
    all_sample_ids = Duke.sample_ids()
    excluded_indexes = [
        "029",
        "050",
        "108",
        "122",
        "127",
        "130",
        "138",
        "151",
        "154",
        "155",
        "159",
        "162",
        "171",
        "179",
        "182",
        "194",
        "208",
        "213",
        "222",
        "226",
        "243",
        "248",
        "257",
        "272",
        "276",
        "279",
        "302",
        "309",
        "314",
        "332",
        "347",
        "359",
        "367",
        "382",
        "388",
        "391",
        "406",
        "422",
        "434",
        "447",
        "449",
        "470",
        "524",
        "549",
        "553",
        "555",
        "571",
        "579",
        "600",
        "619",
        "621",
        "627",
        "637",
        "638",
        "658",
        "701",
        "719",
        "733",
        "747",
        "775",
        "779",
        "785",
        "810",
        "813",
        "828",
        "837",
        "848",
        "867",
        "918",
        "919",
    ]
    excluded_sample_ids = set(["Breast_MRI_" + s for s in excluded_indexes])

    selected_samples_id = [s for s in all_sample_ids if s not in excluded_sample_ids]
    return selected_samples_id
