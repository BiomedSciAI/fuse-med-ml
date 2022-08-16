from ast import Str
import glob
import os
from shutil import ExecError
from typing import Callable, Optional, Union, List, Dict, Any
import logging

import SimpleITK as sitk
from SimpleITK.SimpleITK import Image
import h5py

import numpy as np
import pydicom
from scipy.ndimage.morphology import binary_dilation

from fuse.data import OpBase, get_sample_id
from fuse.utils import NDict
from typing import Tuple
import cv2
import radiomics


class OpExtractDicomsPerSeq(OpBase):
    """
    Extracts patient dicoms per sequence ID and stores them in the sample dict with a given prefix.
    If sequence doesn't exist for the patient, stores an empty list.

    TODO
    Sagi's thoughts,
    1. It should be called "OpExtractDicomsPerSeqID".
    2. In general the naming "sequence ID" doens't suit well. because each sequence ID can have several sequences...
        Is it a convention? if not, it might be better to call it sequence TYPE !
    3. If we call it TYPE, we should change the naming acordingly.
    4. maybe extract the dicoms per sequence, and in another Op gauther per TYPE!
    """

    def __init__(
        self,
        seq_ids: List[str],
        series_desc_2_sequence_map: Dict[str, Str],
        use_order_indicator: bool = False,
        **kwargs,
    ):
        """
        :param seq_ids: the sequences for which the dicoms will be extract
        :param series_desc_2_sequence_map: TODO
        :param use_order_indicator: TODO
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._seq_ids = seq_ids
        self._series_desc_2_sequence_map = series_desc_2_sequence_map
        self._use_order_indicator = use_order_indicator

    def __call__(self, sample_dict: NDict, key_in: str, key_out_seq_ids: str, key_out_sequence_prefix: str):
        """
        TODO
        :param sample_dict:
        :param key_in: key for the sample's path the stored in the sample_dict
        :param key_out_seq_ids: the ids will be used as a suffix in storing the series in the sample_dict
        :param key_out_sequence_prefix: the prefix used to store the series in the sample_dict
        """
        sample_path = sample_dict[key_in]
        sample_dict[key_out_seq_ids] = []
        seq_2_info_map = extract_seq_2_info_map(sample_path, self._series_desc_2_sequence_map)

        for seq_id in self._seq_ids:
            # If sequence ID doesn't have info list, treat as an empty list
            seq_info_list = seq_2_info_map.get(seq_id, [])
            # OBSOLETE, TODO delete when finish
            # if seq_info_list is None:  # uncomment 3 rows to use michal's
            #     # sequence does not exist for the patient
            #     continue
            sample_dict[key_out_seq_ids].append(seq_id)  # OBSOLETE, TODO delete when finish! it will contain all the seq_ids
            sample_dict[f"{key_out_sequence_prefix}.{seq_id}"] = []

            for seq_info in seq_info_list:  # could be several sequences/series (sequence/series=  path)

                dicom_group_ids, sorted_dicom_groups = sort_dicoms_by_field(
                    seq_path=seq_info["path"],
                    dicom_field=seq_info["dicom_field"],
                    use_order_indicator=self._use_order_indicator,
                )
                for dicom_group_id, dicom_group in zip(dicom_group_ids, sorted_dicom_groups):
                    seq_info2 = dict(
                        path=seq_info["path"],
                        series_num=seq_info["series_num"],
                        series_desc=seq_info["series_desc"],
                        dicoms=dicom_group,  # each sequence/series path may contain several (sub-)sequence/series
                        dicoms_id=dicom_group_id,
                    )
                    sample_dict[f"{key_out_sequence_prefix}.{seq_id}"].append(seq_info2)

        return sample_dict


######################################################################
class OpLoadDicomAsStkVol(OpBase):
    """
    Loads dicoms per sequence, and store it in sequence_info as STK Image
    """

    def __init__(self, seq_ids: List[str], seq_reverse_map=None, **kwargs):
        """
        :param seq_reverse_map: sometimes reverse dicoms orders is needed. the map w
        (for b series in which more than one sequence is provided inside the img_path)
        :param is_file: if True loads all dicoms from img_path
        :param kwargs:
        """
        super().__init__(**kwargs)
        if seq_reverse_map is None:
            seq_reverse_map = {}
        self._seq_reverse_map = seq_reverse_map
        self._seq_ids = seq_ids

    def __call__(self, sample_dict: NDict, key_sequence_prefix: str):
        """
        extract_stk_vol loads dicoms into sitk vol
        :param key_in_seq_ids: key to sample's sequence ids
        :param key_sequence_prefix: prefix for the sequences' infos
        """

        for seq_id in self._seq_ids:
            should_reverse_order = self._seq_reverse_map.get(seq_id, False)

            sequence_info_list = sample_dict[f"{key_sequence_prefix}.{seq_id}"]

            for sequence_info in sequence_info_list:
                # Get STK Image volume for each sequnce instance and add it to the info list
                stk_vol = get_stk_volume(
                    img_path=sequence_info["path"],
                    is_file=False,
                    dicom_files=sequence_info["dicoms"],
                    reverse_order=should_reverse_order,
                )
                sequence_info["stk_volume"] = stk_vol

        return sample_dict


def get_stk_volume(img_path: str, is_file: bool, dicom_files, reverse_order) -> Image:  # Not sure if Image
    """
    TODO

    :param img_path:
    :param is_file:
    :param dicom_files:
    :param reverse_order:
    """
    # load from HDF5
    if img_path[-4::] in "hdf5":
        vol = _read_HDF5_file(img_path)
        return vol

    if is_file:
        vol = sitk.ReadImage(img_path)
        return vol

    series_reader = sitk.ImageSeriesReader()

    if dicom_files is None:
        dicom_files = series_reader.GetGDCMSeriesFileNames(img_path)

    if isinstance(dicom_files, str):
        dicom_files = [dicom_files]
    if img_path not in dicom_files[0]:
        dicom_files = [os.path.join(img_path, dicom_file) for dicom_file in dicom_files]
    dicom_files = dicom_files[::-1] if reverse_order else dicom_files
    series_reader.SetFileNames(dicom_files)
    vol = series_reader.Execute()
    return vol


def _read_HDF5_file(img_path):
    with h5py.File(img_path, "r") as hf:
        _array = np.array(hf["array"])
        _spacing = hf.attrs["spacing"]
        _origin = hf.attrs["origin"]
        _world_matrix = np.array(hf.attrs["world_matrix"])[:3, :3]
        _world_matrix_unit = _world_matrix / np.linalg.norm(_world_matrix, axis=0)
        _world_matrix_unit_flat = _world_matrix_unit.flatten()

    # volume 2 sitk
    vol = sitk.GetImageFromArray(_array)
    vol.SetOrigin([_origin[i] for i in [1, 2, 0]])
    vol.SetDirection(_world_matrix_unit_flat)
    vol.SetSpacing([_spacing[i] for i in [1, 2, 0]])
    return vol


#############################
class OpGroupDCESequences(OpBase):
    """
    Groups all 'DCE_mix_ph' sequences into one 'DCE_mix' sequence

    TODO: Generalize to GroupeSequences
    """

    def __init__(self, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._verbose = verbose

    def __call__(self, sample_dict: NDict, key_seq_ids: str, key_sequence_prefix: str) -> NDict:
        """
        extract_list_of_rel_vol extract the volume per seq based on SER_INX_TO_USE
        and put in one list
        :param sample_dict:
        :param key_seq_ids:
        :param key_sequence_prefix:
        """

        seq_ids = sample_dict[key_seq_ids]

        all_dce_mix_ph_sequences = [f"DCE_mix_ph{i}" for i in range(1, 5)] + ["DCE_mix_ph"]

        existing_dce_mix_ph_sequences = [seq_id for seq_id in all_dce_mix_ph_sequences if seq_id in seq_ids]
        # handle multiphase DCE in different series
        if existing_dce_mix_ph_sequences:
            new_seq_id = "DCE_mix"
            assert new_seq_id not in seq_ids
            seq_ids.append(new_seq_id)
            sample_dict[f"{key_sequence_prefix}.{new_seq_id}"] = []
            for seq_id in existing_dce_mix_ph_sequences:
                sequence_info_list = sample_dict[f"{key_sequence_prefix}.{seq_id}"]
                if seq_id == "DCE_mix_ph":
                    series_num_arr = [a["series_num"] for a in sequence_info_list]
                    inx_sorted = np.argsort(series_num_arr)
                    sequence_info_list = [sequence_info_list[i] for i in inx_sorted]
                sample_dict[f"{key_sequence_prefix}.{new_seq_id}"] += sequence_info_list

                delete_seqeunce_from_dict(
                    seq_id=seq_id,
                    sample_dict=sample_dict,
                    key_sequence_ids=key_seq_ids,
                    key_sequence_prefix=key_sequence_prefix,
                )

        return sample_dict


#############################
class OpSelectVolumes(OpBase):
    """
    Collect volumes according to a selected mri sequences
    """

    def __init__(
        self,
        get_indexes_func,
        selected_seq_ids: list,
        delete_input_volumes: Optional[bool] = False,
        verbose: bool = True,
        **kwargs,
    ):
        """
        :param get_indexes_func:
        :param selected_seq_ids:
        :param delete_input_volumes:
        :param verbose:
        """
        super().__init__(**kwargs)
        self._get_indexes_func = get_indexes_func
        self._selected_seq_ids = selected_seq_ids
        self._delete_input_volumes = delete_input_volumes
        self._verbose = verbose

    def __call__(
        self,
        sample_dict: NDict,
        key_in_seq_ids: str,
        key_in_sequence_prefix: str,
        key_out_volumes: str,
        key_out_volumes_info: Optional[str] = None,
    ) -> NDict:
        """
        TODO
        :param sample_dict:
        :param key_in_seq_ids:
        :param key_in_sequence_prefix:
        :param key_out_volumes:
        :param key_out_volumes_info:
        """

        sample_id = get_sample_id(sample_dict)
        seq_ids = sample_dict[key_in_seq_ids]

        sample_dict[f"{key_out_volumes}"] = []
        sample_dict[f"{key_out_volumes_info}"] = []

        for selected_seq_id in self._selected_seq_ids:

            if selected_seq_id in seq_ids:
                sequence_info_list = sample_dict[f"{key_in_sequence_prefix}.{selected_seq_id}"]
            else:
                sequence_info_list = []

            vol_inx_to_use = _get_as_list(
                self._get_indexes_func(sample_id, selected_seq_id)
            )  # TODO, suplly the indices outside
            for inx in vol_inx_to_use:

                if len(sequence_info_list) == 0:
                    if len(sample_dict[key_out_volumes]) == 0:
                        print("============XXXXXXX", sample_id)  # todo: write into log file
                        lgr = logging.getLogger("Fuse")
                        lgr.info(f"OpSelectVolumes: {sample_id} is excluded", {"attrs": ["bold"]})
                        return None
                    seq_volume_template = sample_dict[key_out_volumes][0]
                    stk_volume = get_zeros_vol(seq_volume_template)
                    selected_sequence_info = {}

                else:
                    if inx >= len(sequence_info_list):
                        inx = -1  # take the last

                    # Get sequence's info and volume
                    selected_sequence_info = sequence_info_list[inx]
                    stk_volume = selected_sequence_info["stk_volume"]

                    # Handle problem reading the volume
                    if len(stk_volume) == 0:
                        seq_volume_template = sample_dict[key_out_volumes][0]
                        stk_volume = get_zeros_vol(seq_volume_template)
                        if self._verbose:
                            print(f"\n - problem with reading {selected_seq_id} volume!")

                # Add info and volume to the result with matching index
                sample_dict[key_out_volumes].append(stk_volume)
                sample_dict[key_out_volumes_info].append(
                    dict(
                        seq_id=selected_seq_id,
                        series_desc=selected_sequence_info.get("series_desc", "NAN"),
                        path=selected_sequence_info.get("path", "NAN"),
                        dicoms_id=selected_sequence_info.get("dicoms_id", "NAN"),
                    )
                )

        if self._delete_input_volumes:
            # To save space and also when caching sample_dict
            for seq_id in seq_ids:
                for sequence_info in sample_dict[f"{key_in_sequence_prefix}.{seq_id}"]:
                    del sequence_info["stk_volume"]

        return sample_dict


# def store(sample_dict):
#     vv = [sitk.GetArrayFromImage(v) for v in sample_dict['data.input.selected_volumes']]
#     vv2 = file_io.load_pickle('/tmp/fuse_temp.pkl')
#     for i in range(len(vv)):
#         # print(vv[i].shape == vv2[i].shape)
#         print(vv[i].max(), vv2[i].max(), np.abs(vv[i]-vv2[i]).max())
############################


class OpResampleStkVolsBasedRef(OpBase):
    """
    Resample selected volumes based on a given reference one.
    TODO: write more clearly (?)
    """

    def __init__(self, reference_inx: int, interpolation: str, **kwargs):
        """
        :param reference_inx: index for the volume that will be used as a reference
        :param interpolation: interpolation mode from - ["linear","nn" (nearest neighbor),"bspline"]
        :param kwargs:
        """
        super().__init__(**kwargs)
        assert reference_inx is not None  # todo: redundant??
        self._reference_inx = reference_inx
        self._interpolation = interpolation

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        """
        :param sample_dict: sample's dict
        :param key: a key to the volumes inside the sample_dict
        """
        # create resampling operator based on the reference volume
        volumes = sample_dict[key]
        assert len(volumes) > 0
        # OBSOLETE CODE TODO delete (?)
        # if self.reference_inx > 0:
        #     volumes = [volumes[self.reference_inx]]+ volumes[:self.reference_inx]+ volumes[volumes+1:]

        # cast volumes to match
        seq_volumes_resampled = [sitk.Cast(v, sitk.sitkFloat32) for v in volumes]
        ref_volume = volumes[self._reference_inx]

        # create the resample operator that will excute the resampling
        resample = self.create_resample(ref_volume, size=ref_volume.GetSize(), spacing=ref_volume.GetSpacing())

        for i in range(len(seq_volumes_resampled)):
            # skip on the referenced volume
            if i == self._reference_inx:  # TODO: change to something more readable (?)
                continue
            seq_volumes_resampled[i] = resample.Execute(seq_volumes_resampled[i])

        sample_dict[key] = seq_volumes_resampled
        return sample_dict

    def create_resample(
        self,
        vol_ref: sitk.sitkFloat32,
        size: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
    ) -> sitk.ResampleImageFilter:
        """
        Creates resample operator

        TODO: this function exists twice in this file! solve it.

        :param vol_ref: sitk vol to use as a ref
        :param size: in pixels ()
        :param spacing: in mm ()
        :return: resample sitk operator
        """

        if self._interpolation == "linear":
            interpolator = sitk.sitkLinear
        elif self._interpolation == "nn":
            interpolator = sitk.sitkNearestNeighbor
        elif self._interpolation == "bspline":
            interpolator = sitk.sitkBSpline
        else:
            raise Exception(
                f"Error: unexpected interpolation mode: {self._interpolation}"
            )  # TODO check this in the constructor??

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(vol_ref)
        resample.SetOutputSpacing(spacing)
        resample.SetInterpolator(interpolator)
        resample.SetSize(size)
        return resample


#######


class OpStackList4DStk(OpBase):
    """
    Stack list of 3D STK Images into a single 4D STK Image while keeping the common meta-data from the referenced Image.
    """

    def __init__(self, delete_input_volumes: bool = False, metadata_ref_idx: int = 0, **kwargs: Any):
        """
        :param delete_input_volumes: set 'True' to deleted the input volumes
        :param metadata_ref_idx: an index for an STK Image in the list to be used as a reference for the metadata
        :param kwargs: see super class
        """
        super().__init__(**kwargs)
        self._metadata_ref_idx = metadata_ref_idx
        self._delete_input_volumes = delete_input_volumes

    def __call__(self, sample_dict: NDict, key_in: str, key_out_volume4d: str, key_out_ref_volume: str):
        """
        :param sample_dict:
        :param key_in: a key for the sample's 3D volumes
        :param key_out_volume4d: a key to output the 4D volume
        :param key_out_ref_volume: a key to output the 3D volume
        """
        vols_stk_list = sample_dict[key_in]
        if self._delete_input_volumes:
            del sample_dict[key_in]

        # Convert 3D images into ndarray and stack them in the last axis
        vol_arr = [sitk.GetArrayFromImage(vol) for vol in vols_stk_list]
        vol_final = np.stack(vol_arr, axis=-1)

        # Convert the 4D array into an image and retrive the info from the referenced image
        vol_final_sitk = sitk.GetImageFromArray(vol_final, isVector=True)
        vol_final_sitk.CopyInformation(vols_stk_list[self._metadata_ref_idx])  # info = common meta-data

        sample_dict[key_out_volume4d] = vol_final_sitk
        sample_dict[key_out_ref_volume] = vols_stk_list[self._metadata_ref_idx]
        return sample_dict


class OpRescale4DStk(OpBase):
    """
    TODO
    """

    def __init__(self, thres: Optional[tuple] = (1.0, 99.0), method: Optional[str] = "noclip", **kwargs) -> None:
        """
        :param thres:
        :param method:
        :param kwargs:
        """
        super().__init__(**kwargs)
        # self._mask_ch_inx = mask_ch_inx  # TODO delete (?)
        self._thres = thres
        self._method = method

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        """
        :param sample_dict:
        :param key:
        """
        stk_vol_4D = sample_dict[key]

        vol_backup = sitk.Image(stk_vol_4D)
        vol_array = sitk.GetArrayFromImage(stk_vol_4D)
        if len(vol_array.shape) < 4:
            # increase the dimension
            vol_array = vol_array[:, :, :, np.newaxis]
        # vol_array_pre_rescale = vol_array.copy()  # TODO delete (?) - michal's
        vol_array = apply_rescaling(vol_array, thres=self._thres, method=self._method)

        # Convert back to image and retrieve information
        vol_final = sitk.GetImageFromArray(vol_array)  # , isVector=True)
        vol_final.CopyInformation(vol_backup)
        vol_final = sitk.Image(
            vol_final
        )  # TODO delete or keep? not sure if necessary because it is already Image object
        sample_dict[key] = vol_final
        return sample_dict


class OpAddMaskFromBoundingBoxAsLastChannel(OpBase):
    """
    Generate a mask from the lesion BB and append it as a new (last) channel
    """

    def __init__(self, name_suffix: Optional[str] = "", **kwargs):
        """
        TODO

        :param name_suffix:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._name_suffix = name_suffix

    def __call__(self, sample_dict: NDict, key_in_patch_annotations: str, key_in_ref_volume: str, key_volume4D: str):
        """
        TODO

        :param sample_dict:
        :param key_in_patch_annotations:
        :param key_in_ref_volume:
        :param key_volume4D:
        """
        vol_ref = sample_dict[key_in_ref_volume]
        patch_annotations = sample_dict[key_in_patch_annotations]
        vol_4D_stk = sample_dict[key_volume4D]

        # Get the volume as a 4D array
        vol_4D_arr = sitk.GetArrayFromImage(vol_4D_stk)
        if len(vol_4D_arr.shape) == 3:
            vol_4D_arr = np.expand_dims(vol_4D_arr, axis=3)
        assert len(vol_4D_arr.shape) == 4

        # Get a mask for the 4D volume
        bbox_coords = patch_annotations[f"bbox{self._name_suffix}"]
        if isinstance(bbox_coords, str):
            bbox_coords = np.fromstring(bbox_coords[1:-1], dtype=np.int32, sep=",")
        else:
            bbox_coords = np.asarray(bbox_coords)
        mask_3D_arr = extract_mask_from_annotation(vol_ref, bbox_coords)
        mask_4D_arr = np.expand_dims(mask_3D_arr, axis=3)

        # Concat the original 4D volume with the 4D array
        vol_4D_arr_new = np.concatenate((vol_4D_arr, mask_4D_arr), axis=3)  # TODO should be axis=4 / -1? 3 isn't new
        vol_4D_new_stk = sitk.GetImageFromArray(vol_4D_arr_new)
        # TODO what about the Information of the image? seems like 'vol_4D_new_stk' doesn't have the info of 'vol_4D_stk', but still runs it.

        sample_dict[key_volume4D] = vol_4D_new_stk
        return sample_dict


class OpCreatePatchVolumes(OpBase):
    """
    TODO
    """

    def __init__(
        self,
        lesion_shape,
        lesion_spacing,
        pos_key: str = None,
        name_suffix: Optional[str] = "",
        delete_input_volumes=False,
        crop_based_annotation=True,
        **kwargs,
    ):
        """
        TODO
        """
        super().__init__(**kwargs)
        self._lesion_shape = lesion_shape
        self._lesion_spacing = lesion_spacing
        self._name_suffix = name_suffix
        self._delete_input_volumes = delete_input_volumes
        self._crop_based_annotation = crop_based_annotation
        if pos_key is None:
            self._pos_key = f"centroid{name_suffix}"
        else:
            self._pos_key = pos_key

    def __call__(
        self,
        sample_dict: NDict,
        key_in_volume4D: str,
        key_in_ref_volume: str,
        key_in_patch_annotations: str,
        key_out: str,
    ) -> NDict:
        """
        TODO
        """
        vol_ref = sample_dict[key_in_ref_volume]
        vol_4D = sample_dict[key_in_volume4D]
        patch_annotations = sample_dict[key_in_patch_annotations]

        # read original position
        pos_orig = patch_annotations[self._pos_key]
        if isinstance(pos_orig, str):  # TODO take care in this case before storing it
            if pos_orig[0] == "(":
                pos_orig = pos_orig[1:-1]
            if "," in pos_orig:
                sep = ","
            else:
                sep = " "
            pos_orig = np.fromstring(pos_orig, dtype=np.float32, sep=sep)
        else:
            pos_orig = np.asarray(pos_orig)

        # transform to pixel coordinate in ref coords
        pos_vol = np.array(vol_ref.TransformPhysicalPointToContinuousIndex(pos_orig.astype(np.float64)))

        cropped_vol_size = (self._lesion_shape[2], self._lesion_shape[1], self._lesion_shape[0])  # TODO can use perm?
        spacing = (self._lesion_spacing[2], self._lesion_spacing[1], self._lesion_spacing[0])  # TODO can use perm?
        if self._crop_based_annotation:
            vol_cropped = crop_lesion_vol_mask_based(
                vol_4D, pos_vol, vol_ref, size=cropped_vol_size, spacing=spacing, mask_inx=-1, is_use_mask=True
            )
        else:
            vol_cropped = crop_lesion_vol(
                vol_4D, pos_vol, vol_ref, center_slice=pos_vol[2], size=cropped_vol_size, spacing=spacing
            )

        vol_cropped_arr = sitk.GetArrayFromImage(vol_cropped)
        if len(vol_cropped_arr.shape) < 4:
            # fix dimensions in case of one seq
            vol_cropped_arr = vol_cropped_arr[:, :, :, np.newaxis]

        vol_cropped_arr = np.moveaxis(vol_cropped_arr, 3, 0)  # move last dimension (sequences /  mask) to be first

        assert not np.isnan(
            vol_cropped_arr
        ).any()  # todo: need to revisit for cases with nans (currently there are none)

        sample_dict[key_out] = vol_cropped_arr

        if self._delete_input_volumes:
            del sample_dict[key_in_volume4D]
            del sample_dict[key_in_ref_volume]
        return sample_dict


#######


class OpStk2Dict(OpBase):
    """
    Cast SimpleITK Image (volume) into a dictionary with the following items:
        "arr": ndarray of the image
        "origin": volume's origin
        "spacing": volume's spacing
        "direction": volume's direction


    Example of use:
        (OpStk2Dict(), dict(keys=["data.input.volume4D", "data.input.ref_volume"]))
    """

    def __call__(self, sample_dict: NDict, keys: List[str]):
        """
        :param sample_dict:
        :param keys: list of volumes' keys that will be converted to dictionaries.
        """
        for key in keys:
            vol_stk = sample_dict[key]
            d = dict(
                arr=sitk.GetArrayFromImage(vol_stk),
                origin=vol_stk.GetOrigin(),
                spacing=vol_stk.GetSpacing(),
                direction=vol_stk.GetDirection(),
            )
            sample_dict[key] = d

        return sample_dict


class OpDict2Stk(OpBase):
    """
    Cast a dictonary into a SimpleITK Image (volume).
    The dictonary should have the following items:
        "arr": ndarray of the image
        "origin": volume's origin
        "spacing": volume's spacing
        "direction": volume's direction

    Example of use:
        (OpDict2Stk(), dict(keys=["data.input.volume4D", "data.input.ref_volume"]))
    """

    def __call__(self, sample_dict: NDict, keys: list) -> NDict:
        """
        TODO

        :param sample_dict:
        :param keys:
        """
        for key in keys:
            d = sample_dict[key]
            vol_stk = sitk.GetImageFromArray(d["arr"])
            vol_stk.SetOrigin(d["origin"])
            vol_stk.SetSpacing(d["spacing"])
            vol_stk.SetDirection(d["direction"])
            sample_dict[key] = vol_stk

        return sample_dict


class OpFixProstateBSequence(OpBase):
    def __call__(
        self,
        sample_dict: NDict,
        op_id: Optional[str],
        key_sequence_ids: str,
        key_path_prefix: str,
        key_in_volumes_prefix: str,
    ):
        seq_ids = sample_dict[key_sequence_ids]
        if "b_mix" in seq_ids:

            B_SER_FIX = [
                "diffusie-3Scan-4bval_fs",
                "ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen",
                "diff tra b 50 500 800 WIP511b alle spoelen",
            ]

            def get_single_item(a):
                if isinstance(a, list):
                    assert len(a) == 1
                    return a[0]
                return a

            b_path = get_single_item(sample_dict[f"{key_path_prefix}b"])

            if os.path.basename(b_path) in B_SER_FIX:
                adc_volume = get_single_item(sample_dict[f"{key_in_volumes_prefix}ADC"])

                for b_seq_id in ["b800", "b400"]:
                    volume = get_single_item(sample_dict[f"{key_in_volumes_prefix}.{b_seq_id}"])

                    volume.CopyInformation(adc_volume)
        return sample_dict


######################################3


class OpDeleteSequences(OpBase):
    def __init__(self, sequences_to_delete, **kwargs):
        super().__init__(**kwargs)
        self._sequences_to_delete = sequences_to_delete

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key_sequence_ids):
        for seq_id in self._sequences_to_delete:
            delete_seqeunce_from_dict(seq_id=seq_id, sample_dict=sample_dict, key_sequence_ids=key_sequence_ids)


def delete_seqeunce_from_dict(seq_id: str, sample_dict: NDict, key_sequence_ids, key_sequence_prefix) -> None:
    """
    Deletes sequence from the sample dict
    :param seq_id:
    :param sample_dict:
    :param key_sequence_ids:
    :param key_sequence_prefix:
    """
    seq_ids = sample_dict[key_sequence_ids]
    if seq_id in seq_ids:
        seq_ids.remove(seq_id)
        del sample_dict[f"{key_sequence_prefix}.{seq_id}"]


class OpRenameSequence(OpBase):
    """
    TODO
    """
    def __init__(self, seq_ids: List[str]):
        """
        :paran seq_ids:
        """
        super().__init__()
        self._seq_ids = seq_ids

    def __call__(self, sample_dict: NDict, seq_id_old: str, seq_id_new: str, key_sequence_prefix: str) -> NDict:
        """
        
        :param seq_id_old:
        :param seq_id_new:
        :param key_sequence_prefix:
        """

        if seq_id_old not in self._seq_ids:
            raise Exception(f"old sequence id ({seq_id_old}) must be in sequence ids ({self._seq_ids}).")

        if seq_id_new not in self._seq_ids:
            raise Exception(f"new sequence id ({seq_id_new}) must be in sequence ids ({self._seq_ids}).")

        if sample_dict[f"{key_sequence_prefix}.{seq_id_new}"] != []:
            raise Exception(f"Cant rename {seq_id_old} to {seq_id_new} because {seq_id_new} already has data.")

        sample_dict[f"{key_sequence_prefix}.{seq_id_new}"] = sample_dict[f"{key_sequence_prefix}.{seq_id_old}"]
        sample_dict[f"{key_sequence_prefix}.{seq_id_old}"] = []
        
        return sample_dict


def rename_seqeunce_from_dict(sample_dict, seq_id_old, seq_id_new, key_sequence_prefix, key_seq_ids):
    """
    TODO func not in use. delete (?). replaced by: OpRenameSequence
    """
    seq_ids = sample_dict[key_seq_ids]
    if seq_id_old in seq_ids:
        # assert seq_id_new not in seq_ids   # michal's before I chang
        assert sample_dict[f"{key_sequence_prefix}.{seq_id_new}"] == [] # New must be empty

        sample_dict[f"{key_sequence_prefix}.{seq_id_new}"] = sample_dict[f"{key_sequence_prefix}.{seq_id_old}"]
        sample_dict[f"{key_sequence_prefix}.{seq_id_old}"] = []
        # del sample_dict[f"{key_sequence_prefix}.{seq_id_old}"]
        # seq_ids.remove(seq_id_old)   # michal's before I changed

    return sample_dict


############################


def get_zeros_vol(vol: Image) -> Image:
    """
    Returns an Image volume full of zeros with the same shape as the refernced one

    :param vol: reference volume that will determine the shape of the output Image.
    """

    # TODO why we need the cast?
    if vol.GetNumberOfComponentsPerPixel() > 1:
        ref_zeros_vol = sitk.VectorIndexSelectionCast(vol, 0)
    else:
        ref_zeros_vol = vol
    zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
    zeros_vol = sitk.GetImageFromArray(zeros_vol)
    zeros_vol.CopyInformation(ref_zeros_vol)
    return zeros_vol


def crop_lesion_vol_mask_based(
    vol: sitk.sitkFloat32,
    position: tuple,
    ref: sitk.sitkFloat32,
    size: Tuple[int, int, int] = (160, 160, 32),
    spacing: Tuple[int, int, int] = (1, 1, 3),
    margin: Tuple[int, int, int] = (20, 20, 0),
    mask_inx=-1,
    is_use_mask=True,
):
    """
    crop_lesion_vol crop tensor around position
    :param vol: vol to crop
    :param position: point to crop around
    :param ref: reference volume
    :param size: size in pixels to crop
    :param spacing: spacing to resample the col
    :param center_slice: z coordinates of position
    :param mask_inx: channel index in which mask is located default: last channel
    :param is_use_mask: use mask to define crop bounding box
    :return: cropped volume
    """

    vol_np = sitk.GetArrayFromImage(vol)
    if is_use_mask:

        mask = sitk.GetArrayFromImage(vol)[:, :, :, mask_inx]
        assert set(np.unique(mask)) <= {0, 1}
        mask = mask.astype(int)
        mask_final = sitk.GetImageFromArray(mask)
        mask_final.CopyInformation(ref)

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(mask_final)
        bounding_box = np.array(lsif.GetBoundingBox(1))
        vol_np[:, :, :, mask_inx] = mask
    else:
        bounding_box = np.array(
            [
                int(position[0]) - int(size[0] / 2),
                int(position[1]) - int(size[1] / 2),
                int(position[2]) - int(size[2] / 2),
                size[0],
                size[1],
                size[2],
            ]
        )
    # in z use a fixed number of slices,based on position
    bounding_box[-1] = size[2]
    bounding_box[2] = int(position[2]) - int(size[2] / 2)

    bounding_box_size = bounding_box[3:5][np.argmax(bounding_box[3:5])]
    dshift = bounding_box[3:5] - bounding_box_size
    dshift = np.append(dshift, 0)

    ijk_min_bound = np.maximum(bounding_box[0:3] + dshift - margin, 0)
    ijk_max_bound = np.maximum(
        bounding_box[0:3] + dshift + [bounding_box_size, bounding_box_size, bounding_box[-1]] + margin, 0
    )

    vol_np_cropped = vol_np[
        ijk_min_bound[2] : ijk_max_bound[2], ijk_min_bound[1] : ijk_max_bound[1], ijk_min_bound[0] : ijk_max_bound[0], :
    ]
    vol_np_resized = np.zeros((size[2], size[0], size[1], vol_np_cropped.shape[-1]))
    for si in range(vol_np_cropped.shape[0]):
        for ci in range(vol_np_cropped.shape[-1]):
            vol_np_resized[si, :, :, ci] = cv2.resize(
                vol_np_cropped[si, :, :, ci], (size[0], size[1]), interpolation=cv2.INTER_AREA
            )

    img = sitk.GetImageFromArray(vol_np_resized)
    return img


def crop_lesion_vol(
    vol: sitk.sitkFloat32,
    position: Tuple[float, float, float],
    ref: sitk.sitkFloat32,
    size: Tuple[int, int, int] = (160, 160, 32),
    spacing: Tuple[int, int, int] = (1, 1, 3),
    center_slice=None,
):
    """
     crop_lesion_vol crop tensor around position
    :param vol: vol to crop
    :param position: point to crop around
    :param ref: reference volume
    :param size: size in pixels to crop
    :param spacing: spacing to resample the col
    :param center_slice: z coordinates of position
    :return: cropped volume
    """

    def get_lesion_mask(position, ref):
        mask = np.zeros_like(sitk.GetArrayViewFromImage(ref), dtype=np.uint8)

        coords = np.round(position[::-1]).astype(int)
        mask[coords[0], coords[1], coords[2]] = 1
        mask = binary_dilation(mask, np.ones((3, 5, 5))) + 0
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.CopyInformation(ref)

        return mask_sitk

    def create_resample(
        vol_ref: sitk.sitkFloat32, interpolation: str, size: Tuple[int, int, int], spacing: Tuple[float, float, float]
    ):
        """
        create_resample create resample operator
        :param vol_ref: sitk vol to use as a ref
        :param interpolation:['linear','nn','bspline']
        :param size: in pixels ()
        :param spacing: in mm ()
        :return: resample sitk operator
        """

        if interpolation == "linear":
            interpolator = sitk.sitkLinear
        elif interpolation == "nn":
            interpolator = sitk.sitkNearestNeighbor
        elif interpolation == "bspline":
            interpolator = sitk.sitkBSpline

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(vol_ref)
        resample.SetOutputSpacing(spacing)
        resample.SetInterpolator(interpolator)
        resample.SetSize(size)
        return resample

    def apply_resampling(
        img: sitk.sitkFloat32,
        mask: sitk.sitkFloat32,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 3),
        size: Tuple[int, int, int] = (160, 160, 32),
        transform: sitk = None,
        interpolation: str = "bspline",
        label_interpolator: sitk = sitk.sitkLabelGaussian,
    ):

        ref = img if img != [] else mask
        size = [int(s) for s in size]
        resample = create_resample(ref, interpolation, size=size, spacing=spacing)

        if ~(transform is None):
            resample.SetTransform(transform)
        img_r = resample.Execute(img)

        resample.SetInterpolator(label_interpolator)
        mask_r = resample.Execute(mask)

        return img_r, mask_r

    mask = get_lesion_mask(position, ref)

    vol.SetOrigin((0,) * 3)
    mask.SetOrigin((0,) * 3)
    vol.SetDirection(np.eye(3).flatten())
    mask.SetDirection(np.eye(3).flatten())

    ma_centroid = mask > 0.5
    label_analysis_filer = sitk.LabelShapeStatisticsImageFilter()
    label_analysis_filer.Execute(ma_centroid)
    centroid = label_analysis_filer.GetCentroid(1)
    offset_correction = np.array(size) * np.array(spacing) / 2
    corrected_centroid = np.array(centroid)
    corrected_centroid[2] = center_slice * np.array(spacing[2])
    offset = corrected_centroid - np.array(offset_correction)

    translation = sitk.TranslationTransform(3, offset)
    img, mask = apply_resampling(vol, mask, spacing=spacing, size=size, transform=translation)

    return img


def extract_mask_from_annotation(vol_ref, bbox_coords):
    """
    TODO: there is a double work in 'OpExtractPatchAnotations' (we also calculate there the mask)
          - fix it
    """
    xstart = bbox_coords[0]
    ystart = bbox_coords[1]
    zstart = bbox_coords[2]
    xsize = bbox_coords[3]
    ysize = bbox_coords[4]
    zsize = bbox_coords[5]

    mask = get_zeros_vol(vol_ref)
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np[zstart : zstart + zsize, ystart : ystart + ysize, xstart : xstart + xsize] = 1.0
    return mask_np


def apply_rescaling(img: np.ndarray, thres: tuple = (1.0, 99.0), method: str = "noclip") -> np.ndarray:
    """
    Rescale each channal using method

    :param img:
    :param thres:
    :param method: method of rescaling,
            "clip" -
            "noclip" -
            "mean" -
            "median" -
    :return:
    """
    eps = 0.000001

    def rescale_single_channel_image(img: np.ndarray) -> np.ndarray:
        """
        Rescale a single channel image (2 or 3 dimensions)
        """
        # Deal with negative values first
        # print(f"DEBUG - shape = {img.shape}")
        min_value = np.min(img)
        if min_value < 0:
            img -= min_value

        if method == "clip":
            val_l, val_h = np.percentile(img, thres)
            img2 = img
            img2[img < val_l] = val_l
            img2[img > val_h] = val_h
            img2 = (img2.astype(np.float32) - val_l) / (val_h - val_l + eps)

        elif method == "mean":
            img2 = img / max(np.mean(img), 1)

        elif method == "median":
            img2 = img / max(np.median(img), 1)
            # write as op TODO (???)
            ######################

        elif method == "noclip":
            val_l, val_h = np.percentile(img, thres)
            img2 = img
            img2 = (img2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        else:
            # TODO raise exception (?)
            img2 = img
        return img2

    # fix outlier image values
    img[np.isnan(img)] = 0
    # Process each channel independently
    if len(img.shape) == 4:
        for i in range(img.shape[-1]):
            img[..., i] = rescale_single_channel_image(img[..., i])
    else:
        img = rescale_single_channel_image(img)

    return img


def extract_seq_2_info_map(sample_path: str, series_desc_2_sequence_map: Dict[str, str]):
    """
    Given patient mri path, returns a dictonary that maps 'sequence ID -> sequence's info list'.
    That's because each patient can have several different sequences (scans) with the same sequence ID.

    sequence's info list contains dictonaries with the following values:
                path -> sequence's path
                series_num -> sequence's series number
                dicom_field -> sequence DICOMs' field
                series_desc -> sequence's series number

    :param sample_path: patient's mri path
    :param series_desc_2_sequence_map: a mapping between series descriptions to sequence ID,
                                       for an instance:  "t2_tse_tra" -> "T2"
    """
    seq_2_info_dict = {}

    # Iterate on each sequence in patient mri folder
    for seq_dir in os.listdir(sample_path):
        seq_path = os.path.join(sample_path, seq_dir)

        # Get all dcm files paths
        dcm_files = glob.glob(os.path.join(seq_path, "*.dcm"))
        if len(dcm_files) == 0:
            raise Exception(f"Could not find any dicoms in the sequence directory: {seq_path} .")

        # Read one of the dicoms dataset
        dcm_ds = pydicom.dcmread(dcm_files[0])  

        # Read series description from one of the dicoms
        series_desc = dcm_ds.SeriesDescription

        # Derive the sequence ID from the series description
        seq_id = series_desc_2_sequence_map.get(series_desc, "UNKNOWN")

        series_num = extract_series_num(dcm_ds)  # TODO maybe take those as input (?)
        dicom_field = extract_dicom_field(dcm_ds, seq_id)

        # Add info to the current sequence ID
        seq_2_info_dict.setdefault(seq_id, []).append(
            dict(path=seq_path, series_num=series_num, dicom_field=dicom_field, series_desc=series_desc)
        )

    return seq_2_info_dict


def extract_series_num(dcm_ds: Union[pydicom.FileDataset, pydicom.dicomdir.DicomDir]) -> int:
    """
    Returns DICOM's series number

    :param dcm_ds: dicom's dataset
    """
    if hasattr(dcm_ds, "AcquisitionNumber"):
        return int(dcm_ds.AcquisitionNumber)
    return int(dcm_ds.SeriesNumber)


def extract_dicom_field(
    dcm_ds: Union[pydicom.FileDataset, pydicom.dicomdir.DicomDir], seq_id: str
) -> Union[Tuple[int, int], None]:
    """
    Returns DICOM's field

    :param dcm_ds: dicom's dataset
    :param seq_id: sequence id
    """
    # dicom key
    if seq_id in ("b_mix", "b"):
        if "DiffusionBValue" in dcm_ds:
            dicom_field = (0x0018, 0x9087)  # 'DiffusionBValue'
        else:
            dicom_field = (0x19, 0x100C)  # simens bval tag 0x19 0x100c
    elif "DCE" in seq_id:
        if "TemporalPositionIdentifier" in dcm_ds:
            dicom_field = (0x0020, 0x0100)  # Temporal Position Identifier
        elif "TemporalPositionIndex" in dcm_ds:
            dicom_field = (0x0020, 0x9128)  # Temporal Position Index
        else:
            dicom_field = (0x0020, 0x0012)  # Acqusition Number
    elif seq_id == "MASK":
        dicom_field = (0x0020, 0x0011)  # series number
    else:
        dicom_field = None

    return dicom_field


def sort_dicoms_by_field(seq_path: str, dicom_field, use_order_indicator):
    """
    Return location dir of requested sequence

    sort_dicom_by_dicom_field sorts the dcm_files based on dicom_field
    For some MRI sequences different kinds of MRI series are mixed together (as in bWI) case
    This function creates a dict={dicom_field_type:list of relevant dicoms},
    than concats all to a list of the different series types

    :param dcm_files: list of all dicoms , mixed
    :param dicom_field: dicom field to sort based on
    :return: sorted_names_list, list of sorted dicom series
    """
    dcm_files = glob.glob(os.path.join(seq_path, "*.dcm"))
    dcm_values = {}
    dcm_patient_z = {}
    dcm_instance = {}

    dcm_ds_list = [pydicom.dcmread(dcm) for dcm in dcm_files]
    n_unique_z_image_position_patient = len(np.unique([dcm_ds.ImagePositionPatient[2] for dcm_ds in dcm_ds_list]))
    for index, (dcm, dcm_ds) in enumerate(zip(dcm_files, dcm_ds_list)):
        patient_z = int(dcm_ds.ImagePositionPatient[2])
        instance_num = int(dcm_ds.InstanceNumber)
        if dicom_field is not None:
            val = int(dcm_ds[dicom_field].value)
        else:
            # sort by
            val = int(np.floor((instance_num - 1) / n_unique_z_image_position_patient))

        if val not in dcm_values:
            dcm_values[val] = []
            dcm_patient_z[val] = []
            dcm_instance[val] = []
        dcm_values[val].append(os.path.split(dcm)[-1])
        dcm_patient_z[val].append(patient_z)
        dcm_instance[val].append(instance_num)

    # ex-sort sub-sequences
    sorted_keys = np.sort(list(dcm_values.keys()))
    sorted_names_list = [dcm_values[key] for key in sorted_keys]
    dcm_patient_z_list = [dcm_patient_z[key] for key in sorted_keys]
    dcm_instance_list = [dcm_instance[key] for key in sorted_keys]

    # in-sort each sub-sequence
    if use_order_indicator:
        # sort from low patient z to high patient z
        sorted_names_list_ = [
            list(np.array(list_of_names)[np.argsort(list_of_z)])
            for list_of_names, list_of_z in zip(sorted_names_list, dcm_patient_z_list)
        ]
    else:
        # sort by instance number
        sorted_names_list_ = [
            list(np.array(list_of_names)[np.argsort(list_of_z)])
            for list_of_names, list_of_z in zip(sorted_names_list, dcm_instance_list)
        ]

    return sorted_keys, sorted_names_list_


def _get_as_list(x):
    if isinstance(x, list):
        return x
    return [x]


############################
class OpExtractLesionPropFromBBoxAnotation(OpBase):
    def __init__(self, get_annotations_func, **kwargs):
        super().__init__(**kwargs)
        self._get_annotations_func = get_annotations_func

    def __call__(self, sample_dict: NDict, key_in_ref_volume: str, key_out_lesion_prop: str, key_out_cols: str):
        sample_id = get_sample_id(sample_dict)
        annotations_df = self._get_annotations_func(sample_id)

        vol_ref = sample_dict[key_in_ref_volume]
        sample_dict[key_out_lesion_prop] = []

        bbox_coords = (
            (
                annotations_df[annotations_df["Patient ID"] == sample_id]["Start Column"].values[0],
                annotations_df[annotations_df["Patient ID"] == sample_id]["Start Row"].values[0],
            ),
            (
                annotations_df[annotations_df["Patient ID"] == sample_id]["End Column"].values[0],
                annotations_df[annotations_df["Patient ID"] == sample_id]["End Row"].values[0],
            ),
        )

        start_slice = annotations_df[annotations_df["Patient ID"] == sample_id]["Start Slice"].values[0]
        end_slice = annotations_df[annotations_df["Patient ID"] == sample_id]["End Slice"].values[0]
        lesion_prop, cols = extract_lesion_prop_from_annotation(vol_ref, bbox_coords, start_slice, end_slice)

        sample_dict[key_out_lesion_prop] = lesion_prop
        sample_dict[key_out_cols] = cols

        return sample_dict


class OpExtractPatchAnotations(OpBase):
    """
    TODO Make the Op more generic or maybe move it to duke.py (?)
    """

    def __call__(self, sample_dict: NDict, key_in_ref_volume: str, key_in_annotations, key_out: str):
        """
        TODO

        :param sample_dict:
        :param key_in_ref_volume:
        :param key_in_annotations:
        :key_out:
        """
        vol_ref = sample_dict[key_in_ref_volume]
        annotations = sample_dict[key_in_annotations]

        lesion_prop, cols = extract_lesion_prop_from_annotation(
            vol_ref=vol_ref,
            start_slice=annotations["Start Slice"],
            end_slice=annotations["End Slice"],
            start_row=annotations["Start Row"],
            end_row=annotations["End Row"],
            start_column=annotations["Start Column"],
            end_column=annotations["End Column"],
        )

        sample_dict[key_out] = dict(zip(cols, lesion_prop[0]))

        return sample_dict


def extract_lesion_prop_from_mask(mask, T_connecetd_component_dist=40, minimumObjectSize=3):
    """
    Extract lesion properties from the lesion mask

    :param mask:
    :param T_connecetd_component_dist:
    :param minimumObjectSize:
    """
    mask_bool = mask > 0

    dist_img = sitk.SignedMaurerDistanceMap(
        mask_bool, insideIsPositive=False, squaredDistance=False, useImageSpacing=False
    )
    seeds = sitk.ConnectedComponent(dist_img < T_connecetd_component_dist)
    seeds = sitk.RelabelComponent(seeds, minimumObjectSize=minimumObjectSize)
    ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)  # slow...
    ws = sitk.Mask(ws, sitk.Cast(mask_bool, ws.GetPixelID()))

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(ws)
    stats_list = [
        (
            shape_stats.GetCentroid(i),
            shape_stats.GetBoundingBox(i),
            shape_stats.GetPhysicalSize(i),
            shape_stats.GetElongation(i),
            shape_stats.GetOrientedBoundingBoxSize(i)[0],
            shape_stats.GetOrientedBoundingBoxSize(i)[1],
            shape_stats.GetOrientedBoundingBoxSize(i)[2],
            shape_stats.GetRoundness(i),
            max(shape_stats.GetEquivalentEllipsoidDiameter(i)),
            shape_stats.GetFlatness(i),
        )
        for i in shape_stats.GetLabels()
    ]

    cols = [
        "centroid",
        "bbox",
        "volume",
        "elongation",
        "size_bbox_x",
        "size_bbox_y",
        "size_bbox_z",
        "roudness",
        "longest_elip_diam",
        "flateness",
    ]
    return stats_list, cols


def extract_lesion_prop_from_annotation(vol_ref, start_slice, end_slice, start_row, end_row, start_column, end_column):
    """
    Extract lesion properties from annotations

    :param vol_ref:
    TODO
    :param start_slice:
    :param end_slice:
    """
    # Retrive an ndarray mask that matches the annotations
    mask = get_zeros_vol(vol_ref)  # TODO there is a double work here. maybe can be optimized.
    mask_np = sitk.GetArrayFromImage(mask)
    # mask_np[start_slice:end_slice, bbox_coords[0][1] : bbox_coords[1][1], bbox_coords[0][0] : bbox_coords[1][0]] = 1.0
    mask_np[start_slice:end_slice, start_row:end_row, start_column:end_column] = 1.0

    # Convert the ndarray into an Image
    mask_final = sitk.GetImageFromArray(mask_np)
    mask_final.CopyInformation(vol_ref)
    mask_final = sitk.Image(mask_final)

    return extract_lesion_prop_from_mask(mask_final)


############################
# radiomics operator


class OpReadSTKImage(OpBase):
    """
    TODO
    """

    def __init__(self, seq_id: str, get_image_file: Callable, **kwargs):
        super().__init__(**kwargs)
        self._seq_id = seq_id
        self._get_image_file = get_image_file

    def __call__(self, sample_dict: NDict, key_sequence_prefix: str, key_seq_ids: str):
        sample_id = get_sample_id(sample_dict)
        img_file = self._get_image_file(sample_id)
        vol = sitk.ReadImage(img_file)
        sample_dict[key_seq_ids].append(self._seq_id)
        sample_dict[f"{key_sequence_prefix}.{self._seq_id}"] = [dict(path=img_file, stk_volume=vol)]
        return sample_dict


class OpExtractRadiomics(OpBase):
    """
    TODO
    """

    def __init__(self, extractor, setting, **kwargs):
        super().__init__(**kwargs)
        self.seq_inx_list = setting["seq_inx_list"]
        self.seq_list = setting["seq_list"]
        self.setting = setting
        self.extractor = extractor

    def __call__(self, sample_dict: NDict, key_in_vol_4d: str, key_out_radiomics_results: str):
        """
        TODO
        """

        vol = sitk.GetArrayFromImage(sample_dict[key_in_vol_4d])
        # fix vol to shape of tensor volume
        vol_np = np.moveaxis(vol, 3, 0)

        sample_dict[key_out_radiomics_results] = []

        maskPath = get_maskpath(vol_np, mask_inx=-1, mask_type=self.setting["maskType"])
        result_all = {}
        for seq_inx, seq in zip(self.seq_inx_list, self.seq_list):
            imagePath = get_imagepath(vol_np, seq_inx)
            if self.setting["norm_method"] != "default":
                imagePath = norm_volume(
                    vol_np, seq_inx, imagePath, maskPath, self.setting, normMethod=self.setting["norm_method"]
                )

            result = self.extractor.execute(imagePath, maskPath)
            keys_ = list(result.keys())
            for key in keys_:
                new_key = key + "_seq=" + seq + "_" + self.setting["maskType"]
                result[new_key] = result.pop(key)
            result_all.update(result)

            if self.setting["applyLog"]:
                sigmaValues = np.arange(5.0, 0.0, -0.5)[::1]
                for logImage, imageTypeName, inputKwargs in radiomics.imageoperations.getLoGImage(
                    imagePath, maskPath, sigma=sigmaValues
                ):
                    logFirstorderFeatures = radiomics.firstorder.RadiomicsFirstOrder(logImage, maskPath, **inputKwargs)
                    logFirstorderFeatures.enableAllFeatures()
                    result = logFirstorderFeatures.execute()
                    keys_ = list(result.keys())
                    for key in keys_:
                        new_key = key + "_seq=" + seq + "_" + self.setting["maskType"]
                        result[new_key] = result.pop(key)
                    result_all.update(result)

            #
            # Show FirstOrder features, calculated on a wavelet filtered image
            #
            if self.setting["applyWavelet"]:
                for decompositionImage, decompositionName, inputKwargs in radiomics.imageoperations.getWaveletImage(
                    imagePath, maskPath
                ):
                    waveletFirstOrderFeaturs = radiomics.firstorder.RadiomicsFirstOrder(
                        decompositionImage, maskPath, **inputKwargs
                    )
                    waveletFirstOrderFeaturs.enableAllFeatures()
                    result = waveletFirstOrderFeaturs.execute()
                    keys_ = list(result.keys())
                    for key in keys_:
                        new_key = key + "_seq=" + seq + "_" + self.setting["maskType"]
                        result[new_key] = result.pop(key)
                    result_all.update(result)

        sample_dict[key_out_radiomics_results] = result_all

        return sample_dict


def norm_volume(vol_np, seq_inx, imagePath, maskPath, setting, normMethod="default", vol_inx_for_breast_seg=6):
    if normMethod == "default":
        return imagePath

    if normMethod == "tumor_area":
        dil_filter = sitk.BinaryDilateImageFilter()
        dil_filter.SetKernelRadius(20)
        maskPath_binary = sitk.Cast(maskPath, sitk.sitkInt8)
        tumor_extand_mask = dil_filter.Execute(maskPath_binary)
        firstorder_tmp = radiomics.firstorder.RadiomicsFirstOrder(imagePath, tumor_extand_mask, **setting)
        firstorder_tmp.enableFeatureByName("Mean", True)
        firstorder_tmp.enableFeatureByName("Variance", True)
        results_tmp = firstorder_tmp.execute()
        print(results_tmp)
        imagePath = (imagePath - results_tmp["Mean"]) / np.sqrt(results_tmp["Variance"])
    # elif normMethod == 'breast_area': #todo: check installation of FCM!!!
    #     vol_shape = vol_np.shape
    #
    #     image = vol_np[vol_inx_for_breast_seg, int(vol_shape[1] / 2), :, :]
    #     image_norm = (image - image.min()) / (image.max() - image.min()) * 256
    #     image_rgb = image_norm.astype(np.uint8)
    #     image_seg = FCM(image=image_rgb, image_bit=8, n_clusters=2, m=2, epsilon=0.05, max_iter=100)  # https://github.com/jeongHwarr/various_FCM_segmentation
    #     image_seg.form_clusters()
    #     image_seg_res = image_seg.segmentImage()
    #     breast_label = 1 - image_seg_res[2, 2]
    #     mask_breast = np.zeros(image_seg_res.shape)
    #     mask_breast[image_seg_res == breast_label] = 1
    #
    #     vol_slice = vol_np[seq_inx, int(vol_shape[1] / 2), :, :]
    #     img_mean = np.mean(vol_slice[mask_breast == 1])
    #     img_std = np.std(vol_slice[mask_breast == 1])
    #     imagePath = (imagePath - img_mean) / img_std
    else:
        raise NotImplementedError(normMethod)

    return imagePath


def get_maskpath(vol_np, mask_inx=-1, mask_type="full"):
    """
    TODO
    """
    maskPath = sitk.GetImageFromArray(vol_np[mask_inx, :, :, :])
    if mask_type == "edge":
        maskPath_binary = sitk.Cast(maskPath, sitk.sitkInt8)
        maskPath_edge = sitk.BinaryDilate(maskPath_binary) - sitk.BinaryErode(maskPath_binary)
        maskPath = maskPath_edge

    return maskPath


def get_imagepath(vol_np: np.ndarray, seq_inx: int):
    """
    TODO
    """
    imagePath = sitk.GetImageFromArray(vol_np[seq_inx, :, :, :])
    return imagePath


##### Sagi's Versions #####

"""
Version of 'OpExtractDicomsPerSeq' aims to make the extraction more explicitly PER SEQUENCE,
"""


class OpExtractDicoms(OpBase):
    """
    Extracts dicoms for a given sequence and stores them in the sample dict with a given prefix
    """

    def __init__(
        self,
        seq_ids: str,
        matching_series_descriptions: Dict[str, Str],
        use_order_indicator: bool = False,
        **kwargs,
    ):
        """
        :param seq_id: the sequence for which the dicoms will be extract
        :param matching_series_descriptions: series descriptions that match to the given sequence
        :param use_order_indicator: TODO
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._seq_ids = seq_ids
        self._matching_series_descriptions = matching_series_descriptions
        self._use_order_indicator = use_order_indicator

    def __call__(self, sample_dict: NDict, key_in: str, key_out_seq_ids: str, key_out_sequence_prefix: str):
        """
        # TODO I need to make another Op to retriev thos key_out_seq_ids for each sample!
        :param sample_dict:
        :param key_in: key for the mri path
        :param key_out_seq_ids: the ids will be used as a suffix in storing the series in the sample_dict
        :param key_out_sequence_prefix: the prefix used to store the series in the sample_dict
        """
        sample_path = sample_dict[key_in]
        sample_dict[key_out_seq_ids] = []
        seq_2_info_map = extract_seq_2_info_map(sample_path, self._series_desc_2_sequence_map)
        # print(f"DEBUG: seq_2_info_map = {seq_2_info_map}")
        for seq_id in self._seq_ids:
            seq_info_list = seq_2_info_map.get(seq_id, None)
            if seq_info_list is None:
                # sequence does not exist for the patient
                continue
            sample_dict[key_out_seq_ids].append(seq_id)
            sample_dict[f"{key_out_sequence_prefix}.{seq_id}"] = []
            for seq_info in seq_info_list:  # could be several sequences/series (sequence/series=  path)

                dicom_group_ids, sorted_dicom_groups = sort_dicoms_by_field(
                    seq_path=seq_info["path"],
                    dicom_field=seq_info["dicom_field"],
                    use_order_indicator=self._use_order_indicator,
                )
                for dicom_group_id, dicom_group in zip(dicom_group_ids, sorted_dicom_groups):
                    seq_info2 = dict(
                        path=seq_info["path"],
                        series_num=seq_info["series_num"],
                        series_desc=seq_info["series_desc"],
                        dicoms=dicom_group,  # each sequence/series path may contain several (sub-)sequence/series
                        dicoms_id=dicom_group_id,
                    )
                    sample_dict[f"{key_out_sequence_prefix}.{seq_id}"].append(seq_info2)

        return sample_dict

    class OpGetSeqIds(OpBase):
        """
        Get the relevant sequence ids per sample
        """

        def __init__(self, seq_2_series_desc_map: Dict[str, str]):
            """
            TODO
            """
            super().__init__()
            self._seq_2_series_desc_map = seq_2_series_desc_map

        def __call__(self, sample_dict: NDict, key_in_path: str, key_out: str) -> NDict:
            """
            TODO
            """
            sample_path = sample_dict[key_in_path]

            res = []
            # Iterate on each sequence in patient mri folder
            for seq_dir in os.listdir(sample_path):
                seq_path = os.path.join(sample_path, seq_dir)

                # Read series description from dcm files
                dcm_files = glob.glob(os.path.join(seq_path, "*.dcm"))  # get all dcm files paths
                # TODO throw an error or a warning if there are no dicoms
                dcm_ds = pydicom.dcmread(dcm_files[0])  # Read one of the dicoms dataset
                # print(f"DEBUG: dcm_files = {dcm_files}")  # TODO delete
                series_desc = dcm_ds.SeriesDescription

                seq_ids = self._seq_2_series_desc_map.keys()
                for seq_id in seq_ids:
                    if series_desc in self._seq_2_series_desc_map[seq_id]:
                        pass  # TODO FINISH

            sample_dict[key_out] = res
            return sample_dict
