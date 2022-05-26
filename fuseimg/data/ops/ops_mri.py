import glob
import os
from typing import Optional

import SimpleITK as sitk
import h5py

import numpy as np
import pydicom

from fuse.data import OpBase, get_sample_id
from fuse.utils import NDict
from typing import Tuple



class OpExtractDicomsPerSeq(OpBase):

    def __init__(self, seq_ids, seq_dict, use_order_indicator: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._seq_ids = seq_ids
        self._seq_dict = seq_dict
        self._use_order_indicator = use_order_indicator



    def __call__(self, sample_dict: NDict, key_in:str,  key_out_sequences:str,
                 key_out_path_prefix:str, key_out_dicoms_prefix: str, key_out_series_num_prefix:str):
        sample_path = sample_dict[key_in]

        for seq_id in self._seq_ids:

            seq_path = self.extract_seq_dir(sample_path, seq_id)
            if seq_path is None:
                # sequence does not exist for the patient
                continue

            if key_out_sequences not in sample_dict:
                sample_dict[key_out_sequences] = []
            sample_dict[key_out_sequences].append(seq_id)

            seq_dicom_path = os.path.join(seq_path, os.listdir(seq_path)[0])
            dcm_ds = pydicom.dcmread(seq_dicom_path)

            series_num = self.extract_ser_num(dcm_ds)
            dicom_field = self.extract_dicom_field(dcm_ds, seq_id)
            sorted_dicom_list = self.sort_dicoms_by_field(seq_path, dicom_field)

            dicom_list = sorted_dicom_list

            sample_dict[f'{key_out_path_prefix}{seq_id}'] = seq_path
            sample_dict[f'{key_out_dicoms_prefix}{seq_id}'] = dicom_list  # seq_dicom_list
            sample_dict[f'{key_out_series_num_prefix}{seq_id}'] = series_num


        return sample_dict

    def extract_seq_dir(self, sample_path, seq_desc):
        seq_dir_files = os.listdir(sample_path)

        for seq_desc in self._seq_dict[seq_desc]:
            match_seq_dirs = [seq for seq in seq_dir_files if seq_desc in seq]
            assert len(match_seq_dirs) <= 1
            if len(match_seq_dirs) > 0:
                return os.path.join(sample_path, match_seq_dirs[0])

        print(f"OpExtractSeqInfo: no {seq_desc} was found under path")
        return None

    def extract_ser_num(self, dcm_ds):

        # series number
        if hasattr(dcm_ds, 'AcquisitionNumber'):
            return int(dcm_ds.AcquisitionNumber)
        return int(dcm_ds.SeriesNumber)

    def extract_dicom_field(self, dcm_ds, seq_desc):

        # dicom key
        if seq_desc == 'b_mix':
            if 'DiffusionBValue' in dcm_ds:
                dicom_field = (0x0018, 0x9087)  # 'DiffusionBValue'
            else:
                dicom_field = (0x19, 0x100c)
        elif 'DCE' in seq_desc:
            if 'TemporalPositionIdentifier' in dcm_ds:
                dicom_field = (0x0020, 0x0100)  # Temporal Position Identifier
            elif 'TemporalPositionIndex' in dcm_ds:
                dicom_field = (0x0020, 0x9128)
            else:
                dicom_field = (0x0020, 0x0012)  # Acqusition Number
        elif seq_desc == 'MASK':
            dicom_field = (0x0020, 0x0011)  # series number
        else:
            dicom_field = 'NAN'

        return dicom_field

    def sort_dicoms_by_field(self, seq_path, dicom_field):
        '''
        Return location dir of requested sequence
        '''

        """
         sort_dicom_by_dicom_field sorts the dcm_files based on dicom_field
         For some MRI sequences different kinds of MRI series are mixed together (as in bWI) case
         This function creates a dict={dicom_field_type:list of relevant dicoms},
         than concats all to a list of the different series types
         :param dcm_files: list of all dicoms , mixed
         :param dicom_field: dicom field to sort based on
         :return: sorted_names_list, list of sorted dicom series
         """
        dcm_files = glob.glob(os.path.join(seq_path, '*.dcm'))
        dcm_values = {}
        dcm_patient_z = {}
        dcm_instance = {}
        for index, dcm in enumerate(dcm_files):
            dcm_ds = pydicom.dcmread(dcm)
            patient_z = int(dcm_ds.ImagePositionPatient[2])
            instance_num = int(dcm_ds.InstanceNumber)
            try:
                val = int(dcm_ds[dicom_field].value)
                if val not in dcm_values:
                    dcm_values[val] = []
                    dcm_patient_z[val] = []
                    dcm_instance[val] = []
                dcm_values[val].append(os.path.split(dcm)[-1])
                dcm_patient_z[val].append(patient_z)
                dcm_instance[val].append(instance_num)
            except:
                # sort by
                if index == 0:
                    patient_z_ = []
                    for dcm_ in dcm_files:
                        dcm_ds_ = pydicom.dcmread(dcm_)
                        patient_z_.append(dcm_ds_.ImagePositionPatient[2])
                val = int(np.floor((instance_num - 1) / len(np.unique(patient_z_))))
                if val not in dcm_values:
                    dcm_values[val] = []
                    dcm_patient_z[val] = []
                    dcm_instance[val] = []
                dcm_values[val].append(os.path.split(dcm)[-1])
                dcm_patient_z[val].append(patient_z)
                dcm_instance[val].append(instance_num)

        sorted_keys = np.sort(list(dcm_values.keys()))
        sorted_names_list = [dcm_values[key] for key in sorted_keys]
        dcm_patient_z_list = [dcm_patient_z[key] for key in sorted_keys]
        dcm_instance_list = [dcm_instance[key] for key in sorted_keys]

        if self._use_order_indicator:
            # sort from low patient z to high patient z
            sorted_names_list_ = [list(np.array(list_of_names)[np.argsort(list_of_z)]) for list_of_names, list_of_z in
                                  zip(sorted_names_list, dcm_patient_z_list)]
        else:
            # sort by instance number
            sorted_names_list_ = [list(np.array(list_of_names)[np.argsort(list_of_z)]) for list_of_names, list_of_z in
                                  zip(sorted_names_list, dcm_instance_list)]

        return sorted_names_list_

######################################################################
class OpLoadDicomAsStkVol(OpBase):
    '''
    Return location dir of requested sequence
    '''

    def __init__(self,reverse_order: bool=False, is_file: bool=False, **kwargs):
        """
        :param reverse_order: sometimes reverse dicoms orders is needed
        (for b series in which more than one sequence is provided inside the img_path)
        :param is_file: if True loads all dicoms from img_path
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._reverse_order = reverse_order
        self._is_file = is_file

    def __call__(self, sample_dict: NDict, key_in_seq_ids:str, key_in_path_prefix: str, key_in_dicoms_prefix: str,
                 key_out_prefix: str, reverse_order: Optional[bool]=None, is_file:Optional[bool]=None):
        """
        extract_stk_vol loads dicoms into sitk vol
        :param img_path: path to dicoms - load all dicoms from this path
        :param img_list: list of dicoms to load
        :return: list of stk vols
        """
        if is_file is None:
            is_file = self._is_file
        if reverse_order is None:
            reverse_order = self._reverse_order

        seq_ids = sample_dict[key_in_seq_ids]

        for seq_id in seq_ids:

            img_path = sample_dict[ f'{key_in_path_prefix}{seq_id}']

            stk_vols = []

            try:
                # load from HDF5
                if img_path[-4::] in 'hdf5':
                    vol = _read_HDF5_file(img_path)
                    stk_vols.append(vol)

                elif is_file:
                    vol = sitk.ReadImage(img_path)
                    stk_vols.append(vol)

                else:
                    series_reader = sitk.ImageSeriesReader()
                    dicom_files = sample_dict[f'{key_in_dicoms_prefix}{seq_id}']

                    if not dicom_files:
                        dicom_files = [series_reader.GetGDCMSeriesFileNames(img_path)]

                    for i_img, imgs_names in enumerate(dicom_files):
                        if isinstance(imgs_names, str):
                            imgs_names = [imgs_names]
                        if img_path not in imgs_names[0]:
                            imgs_names = [os.path.join(img_path, f) for f in imgs_names]
                        dicom_names = imgs_names[::-1] if reverse_order else imgs_names
                        series_reader.SetFileNames(dicom_names)
                        imgs = series_reader.Execute()
                        stk_vols.append(imgs)

                sample_dict[f'{key_out_prefix}{seq_id}'] = stk_vols

            except Exception as e:
                print(e)
        return sample_dict

def _read_HDF5_file(img_path):
    with h5py.File(img_path, 'r') as hf:
        _array = np.array(hf['array'])
        _spacing = hf.attrs['spacing']
        _origin = hf.attrs['origin']
        _world_matrix = np.array(hf.attrs['world_matrix'])[:3, :3]
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
    def __init__(self, verbose: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._verbose = verbose

    def __call__(self, sample_dict: NDict, key_sequence_ids: str,
                 key_volumes_prefix: str, key_path_prefix: str, key_series_num_prefix: str):

        """
        extract_list_of_rel_vol extract the volume per seq based on SER_INX_TO_USE
        and put in one list
        :param vols_dict: dict of sitk vols per seq
        :param seq_info: dict of seq description per seq
        :return:

        """

        seq_ids = sample_dict[key_sequence_ids]

        # delete_seqeunce_from_dict(seq_id='b_mix', sample_dict=sample_dict, seq_ids=seq_ids) #todo: no need to delete if not included. Need to be specified by the user

        all_dce_mix_ph_sequences = [f'DCE_mix_ph{i}' for i in range(1, 5)] + ['DCE_mix_ph']
        # seq_id_exist_func = lambda seq_id: seq_id in seq_ids and len(
        #     sample_dict[f'{key_in_volumes_prefix}{seq_id}']) > 0

        existing_dce_mix_ph_sequences = [seq_id for seq_id in all_dce_mix_ph_sequences if seq_id in seq_ids]
        # handle multiphase DCE in different series
        if existing_dce_mix_ph_sequences:
            new_seq_id = 'DCE_mix'
            key_path = f'{key_path_prefix}{new_seq_id}'
            key_volumes = f'{key_volumes_prefix}{new_seq_id}'
            key_series_num = f'{key_series_num_prefix}{new_seq_id}'

            assert new_seq_id not in seq_ids

            seq_ids.append(new_seq_id)
            sample_dict[key_path] = []
            sample_dict[key_volumes] = []
            sample_dict[key_series_num] = []


            for seq_id in existing_dce_mix_ph_sequences:
                seq_path = _get_as_list(sample_dict[f'{key_path_prefix}{seq_id}'])
                stk_vols = _get_as_list(sample_dict[f'{key_volumes_prefix}{seq_id}'])
                series_num = _get_as_list(sample_dict[f'{key_series_num_prefix}{seq_id}'])

                if seq_id == 'DCE_mix_ph':
                    inx_sorted = np.argsort(series_num)
                    for ser_num_inx in inx_sorted:
                        sample_dict[key_volumes].append(stk_vols[ser_num_inx])
                        sample_dict[key_path].append(seq_path[ser_num_inx])
                        sample_dict[key_series_num].append(series_num[ser_num_inx])
                else:
                    sample_dict[key_path] += seq_path
                    sample_dict[key_volumes] += stk_vols
                    sample_dict[key_series_num] += series_num

                _delete_seqeunce_from_dict(seq_id=seq_id, sample_dict=sample_dict, key_sequence_ids=key_sequence_ids)

        return sample_dict

#############################
class OpSelectVolumes(OpBase):
    def __init__(self, subseq_to_use: list, get_indexes_func, verbose: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._subseq_to_use = subseq_to_use
        self._get_indexes_func = get_indexes_func
        self._verbose = verbose

    def __call__(self, sample_dict: NDict,
                 key_in_volumes_prefix: str, key_in_path_prefix: str,
                 key_out_volumes_prefix: str, key_out_path_prefix: str):
        sample_id =get_sample_id(sample_dict)
        for seq_id in self._subseq_to_use:

            seq_volumes = _get_as_list(sample_dict[f'{key_in_volumes_prefix}{seq_id}'])
            seq_path = _get_as_list(sample_dict[f'{key_in_path_prefix}{seq_id}'])
            vol_inx_to_use = _get_as_list(self._get_indexes_func(sample_id, seq_id))

            sample_dict[f'{key_out_volumes_prefix}{seq_id}'] = []
            sample_dict[f'{key_out_path_prefix}{seq_id}'] = []

            for inx in vol_inx_to_use:
                assert len(seq_volumes) > 0

                if inx > len(seq_volumes) - 1:
                    inx = -1  # take the last


                volume = seq_volumes[inx]
                if len(volume) == 0:
                    volume = _get_zeros_vol(seq_volumes[0])
                    if self._verbose:
                        print(f'\n - problem with reading {seq_id} volume!')
                sample_dict[f'{key_out_volumes_prefix}{seq_id}'].append(volume)
                sample_dict[f'{key_out_path_prefix}{seq_id}'].append(seq_path[inx])


        return sample_dict




############################

class OpResampleStkVolsBasedRef(OpBase):
    def __init__(self, reference_inx: int, interpolation: str, **kwargs):
        super().__init__(**kwargs)
        assert reference_inx is not None  # todo: redundant??
        self.reference_inx = reference_inx
        self.interpolation = interpolation

    def __call__(self, sample_dict: NDict,
                 key_seq_volumes_prefix: str, key_out_prefix: str, key_seq_ids: str):

        # ------------------------
        # create resampling operator based on ref vol
        seq_ids = sample_dict[key_seq_ids]
        ref_seq_id = seq_ids[self.reference_inx]
        ref_seq_volumes = sample_dict[f'{key_seq_volumes_prefix}{ref_seq_id}']
        assert len(ref_seq_volumes) == 1
        ref_seq_volume = sitk.Cast(ref_seq_volumes[0], sitk.sitkFloat32)  # todo: verify

        resample = self.create_resample(ref_seq_volume, self.interpolation, size=ref_seq_volume.GetSize(),
                                        spacing=ref_seq_volume.GetSpacing())

        for i_seq, seq_id in enumerate(seq_ids):
            seq_volumes = sample_dict[f'{key_seq_volumes_prefix}{seq_id}']
            if i_seq == self.reference_inx:
                seq_volumes_resampled = [ref_seq_volume]  # do nothing
            else:
                seq_volumes_resampled = [resample.Execute(sitk.Cast(vol, sitk.sitkFloat32)) for vol in seq_volumes]
            sample_dict[f'{key_out_prefix}{seq_id}'] = seq_volumes_resampled

        return sample_dict

    def create_resample(self, vol_ref: sitk.sitkFloat32, interpolation: str, size: Tuple[int, int, int],
                        spacing: Tuple[float, float, float]):
        """
        create_resample create resample operator
        :param vol_ref: sitk vol to use as a ref
        :param interpolation:['linear','nn','bspline']
        :param size: in pixels ()
        :param spacing: in mm ()
        :return: resample sitk operator
        """

        if interpolation == 'linear':
            interpolator = sitk.sitkLinear
        elif interpolation == 'nn':
            interpolator = sitk.sitkNearestNeighbor
        elif interpolation == 'bspline':
            interpolator = sitk.sitkBSpline

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(vol_ref)
        resample.SetOutputSpacing(spacing)
        resample.SetInterpolator(interpolator)
        resample.SetSize(size)
        return resample
#######

class OpFixProstateBSequence(OpBase):

    def __call__(self, sample_dict: NDict, op_id: Optional[str],
                 key_sequence_ids: str, key_path_prefix:str, key_in_volumes_prefix:str):
        seq_ids = sample_dict[key_sequence_ids]
        if 'b' in seq_ids:

            B_SER_FIX = ['diffusie-3Scan-4bval_fs',
                         'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
                         'diff tra b 50 500 800 WIP511b alle spoelen']

            def get_single_item(a):
                if isinstance(a, list):
                    assert len(a)==1
                    return a[0]
                return a
            b_path = get_single_item(sample_dict[f'{key_path_prefix}b'])

            if os.path.basename(b_path) in B_SER_FIX:
                adc_volume = get_single_item(sample_dict[f'{key_in_volumes_prefix}ADC'])

                for b_seq_id in ['b800', 'b400']:
                    volume = get_single_item(sample_dict[f'{key_in_volumes_prefix}{b_seq_id}'])

                    volume.CopyInformation(adc_volume)
        return sample_dict

######################################3


class OpDeleteSequences(OpBase):
    def __init__(self, sequences_to_delete, **kwargs):
        super().__init__(**kwargs)
        self._sequences_to_delete = sequences_to_delete

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key_sequence_ids):
        for seq_id in self._sequences_to_delete:
            _delete_seqeunce_from_dict(seq_id=seq_id, sample_dict=sample_dict, key_sequence_ids=key_sequence_ids)


def _delete_seqeunce_from_dict(seq_id, sample_dict, key_sequence_ids):
    seq_ids = sample_dict[key_sequence_ids]
    if seq_id in seq_ids:
        seq_ids.remove(seq_id)
        keys_to_delete = [k for k in sample_dict.flatten() if k.endswith(f'.{seq_id}')]
        for key in keys_to_delete:
            del sample_dict[key]


############################

def _get_zeros_vol(vol):
    if vol.GetNumberOfComponentsPerPixel() > 1:
        ref_zeros_vol = sitk.VectorIndexSelectionCast(vol, 0)
    else:
        ref_zeros_vol = vol
    zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
    zeros_vol = sitk.GetImageFromArray(zeros_vol)
    zeros_vol.CopyInformation(ref_zeros_vol)
    return zeros_vol


def _get_as_list(x):
    if isinstance(x, list):
        return x
    return [x]