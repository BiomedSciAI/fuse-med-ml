import os

from fuse.data.ops.op_base import OpBase
from typing import Optional
import numpy as np
from fuse.data.ops.ops_common import OpApplyTypes
# import nibabel as nib
from fuse.utils.ndict import NDict
import pandas as pd
import pydicom
import glob
import h5py
import SimpleITK as sitk
from typing import Tuple
from tqdm import tqdm
from fuse.data.utils.sample import get_sample_id


def main():
    root_data = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
    data_path = os.path.join(root_data, 'Duke-Breast-Cancer-MRI')
    data_metadata_path = os.path.join(root_data, 'metadata.csv')

    seq_ids = ['DCE_mix_ph1', 'DCE_mix_ph3']
    TEST_PATIENT_ID, TEST_STUDY_ID = 'Breast_MRI_900', '01-01-1990-BREASTROUTINE DYNAMICS-51487'
    sample_dict = {}
    sample_dict['data.sample_id'] = (TEST_PATIENT_ID, TEST_STUDY_ID)

    seq_dict, SER_INX_TO_USE, exp_patients, _, _ = process_mri_series(data_metadata_path)

    # ------------------------------------------------MRI processor
    op_extract_dicom_per_seq = OpExtractDicomsPerSeq(dir_path=data_path, use_order_indicator=False)

    op_load_dicom_stk_vol = OpLoadDicomAsStkVol(data_path, reverse_order=False, is_path=False)


    # Load volumes dict
    for seq_desc1 in tqdm(seq_ids):
        sample_dict = op_extract_dicom_per_seq(sample_dict,
                                               seq_desc=seq_desc1,
                                               seq_dict=seq_dict,
                                               )

        sample_dict = op_load_dicom_stk_vol(sample_dict,
                                            key_img_path=f'data.input.sequence_path.{seq_desc1}',
                                            key_img_list=f'data.input.img_list.{seq_desc1}',
                                            key_out=f'data.input.sequence_volumes.{seq_desc1}')  #todo: change to seq_volume_list





    ###########

    # get list


    op = OpMakeListOfVol(ser_inx_to_use=SER_INX_TO_USE, subseq_to_use=['DCE_mix'], exp_patients=exp_patients) #todo: special
    sample_dict = op(sample_dict, key_sequence_volumes_prefix='data.input.sequence_volumes.', 
                     key_sequence_path_prefix='data.input.sequence_path.',
                     key_sequence_series_num_prefix='data.input.sequence_series_num.',
                     key_sequence_ids='data.input.sequence_ids')

    op_resample_stk_vols_based_ref = OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline')
    sample_dict = op_resample_stk_vols_based_ref(sample_dict, key_seq_volumes_prefix=f'data.input.sequence_volumes.',
                                                 key_out_prefix=f'data.input.sequence_volumes_resampled.',
                                                 key_seq_ids='data.input.sequence_ids')



    # op = OpStackListStk(reference_inx=0)
    # sample = op.__call__(sample,sample['vols_list'])
    #
    # op = OpRescale4DStk(mask_ch_inx=None)
    # sample = op.__call__(sample,sample['vol4D'])

    # -------------------------------------------------processor

    a = 1


# put in dataset
def process_mri_series(metadata_path: str):  # specific for DUKE data?
    special_patients = ['Breast_MRI_120', 'Breast_MRI_596']

    seq_to_use = ['DCE_mix_ph1',  #first sequence in the reference volume
                  'DCE_mix_ph2',
                  'DCE_mix_ph3',
                  'DCE_mix_ph4',
                  'DCE_mix',
                  'DCE_mix_ph',
                  'MASK']
    subseq_to_use = ['DCE_mix_ph2', 'MASK']

    SER_INX_TO_USE = {
        'all': {'DCE_mix': [1], 'MASK': [0]},
        'Breast_MRI_120': {'DCE_mix': [2], 'MASK': [0]},
        'Breast_MRI_596': {'DCE_mix': [2], 'MASK': [0]}
    }

    opt_seq = [
        '1st', '1ax', '1Ax', '1/ax',  # todo: why don't add '1/AX'?  check if need to be added
        '2nd', '2ax', '2Ax', '2/ax',
        '3rd', '3ax', '3Ax', '3/ax',
        '4th', '4ax', '4Ax', '4/ax',
    ]
    my_keys = ['DCE_mix_ph1'] * 4 + ['DCE_mix_ph2'] * 4 + ['DCE_mix_ph3'] * 4 + ['DCE_mix_ph4'] * 4

    metadata_df = pd.read_csv(metadata_path)
    all_seq_ids = metadata_df['Series Description'].unique()

    all_seq_ids_slash = [s.replace('ax', '/ax').replace('Ax', '/Ax') for s in all_seq_ids] #todo: maybe we can remove this

    seq_to_use_dict = {}
    for opt_seq_tmp, my_key in zip(opt_seq, my_keys):
        if my_key not in seq_to_use_dict.keys():
            seq_to_use_dict[my_key] = []
        seq_to_use_dict[my_key] += [s for s in all_seq_ids if opt_seq_tmp in s] + [s for s in all_seq_ids_slash
                                                                                   if opt_seq_tmp in s]


    if 'DCE_mix_ph' not in seq_to_use_dict.keys():
        seq_to_use_dict['DCE_mix_ph'] = []
    seq_to_use_dict['DCE_mix_ph'] += ['ax dyn']  #todo: at the end this will be part of DCE_mix

    # todo: MASK in subseq_to_use but not in seq_to_use_dict. why do we need subset_to_use?
    # todo: why do we generate all_seq_ids_slash?  they may appear on disk but not in metadata?
    return seq_to_use_dict, SER_INX_TO_USE, special_patients, seq_to_use, subseq_to_use  # todo: can delete the last two items?





# ------------------------------------------------MRI processor

class OpExtractDicomsPerSeq(OpBase):
    '''
    Return location dir of requested sequence
    '''

    def __init__(self, dir_path, use_order_indicator: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path
        self._use_order_indicator = use_order_indicator

    '''
    :param key_in: the key name in sample_dict that holds the filename
    :param key_out:
    '''

    def __call__(self, sample_dict: NDict, seq_desc: str, seq_dict: dict):
        sample_id = get_sample_id(sample_dict)
        if isinstance(sample_id, str):
            sample_path = os.path.join(self._dir_path, sample_id)
        else:
            sample_path = os.path.join(self._dir_path, *sample_id)

        seq_path = self.extract_seq_dir(sample_path, seq_desc, seq_dict)

        seq_dicom_path = os.path.join(seq_path, os.listdir(seq_path)[0])
        dcm_ds = pydicom.dcmread(seq_dicom_path)

        series_num = self.extract_ser_num(dcm_ds)
        dicom_field = self.extract_dicom_field(dcm_ds, seq_desc)
        sorted_dicom_list = self.sort_dicoms_by_field(seq_path, dicom_field)

        dicom_list = sorted_dicom_list

        sample_dict[f'data.input.sequence_path.{seq_desc}'] = seq_path  #todo: change img to seq
        sample_dict[f'data.input.img_list.{seq_desc}'] = dicom_list  #seq_dicom_list
        sample_dict[f'data.input.sequence_series_num.{seq_desc}'] = series_num
        if 'data.input.sequence_ids' not in sample_dict:
            sample_dict['data.input.sequence_ids'] = []
        sample_dict['data.input.sequence_ids'].append(seq_desc)

        return sample_dict

    def extract_seq_dir(self, sample_path, seq_desc, seq_dict):
        seq_dir_files = os.listdir(sample_path)

        for seq_desc in seq_dict[seq_desc]:
            match_seq_dirs = [seq for seq in seq_dir_files if seq_desc in seq]
            assert len(match_seq_dirs) <= 1
            if len(match_seq_dirs) > 0:
                return os.path.join(sample_path, match_seq_dirs[0])

        raise Exception(f"OpExtractSeqInfo: no {seq_desc} was found under path")

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


class OpLoadDicomAsStkVol(OpBase):
    '''
    Return location dir of requested sequence
    '''

    def __init__(self, dir_path: str, reverse_order: bool=False, is_path: bool=False, **kwargs):  #todo: change to is_file
        """

        :param dir_path:
        :param reverse_order: sometimes reverse dicoms orders is needed
        (for b series in which more than one sequence is provided inside the img_path)
        :param is_path: if True loads all dicoms from img_path
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._dir_path = dir_path
        self._reverse_order = reverse_order
        self._is_path = is_path

    def __call__(self, sample_dict: NDict, key_img_path: str, key_img_list: str, key_out: str): #todo: add optional paramter to override is_path and reverse_order
        """
        extract_stk_vol loads dicoms into sitk vol
        :param img_path: path to dicoms - load all dicoms from this path
        :param img_list: list of dicoms to load
        :return: list of stk vols
        """
        img_path = sample_dict[key_img_path]
        img_list = sample_dict[key_img_list]#todo: change to seq_dicom_list

        stk_vols = []

        try:
            # load from HDF5
            if img_path[-4::] in 'hdf5':
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
                stk_vols.append(vol)

            elif self._is_path:
                vol = sitk.ReadImage(img_path)
                stk_vols.append(vol)

            else:
                series_reader = sitk.ImageSeriesReader()

                if img_list == []:
                    img_list = [series_reader.GetGDCMSeriesFileNames(img_path)]

                for i_img, imgs_names in enumerate(img_list):
                    if isinstance(imgs_names, str):
                        imgs_names = [imgs_names]
                    if img_path not in imgs_names[0]:
                        imgs_names = [os.path.join(img_path, f) for f in imgs_names]
                    dicom_names = imgs_names[::-1] if self._reverse_order else imgs_names
                    series_reader.SetFileNames(dicom_names)
                    imgs = series_reader.Execute()
                    stk_vols.append(imgs)

            sample_dict[key_out] = stk_vols
            return sample_dict

        except Exception as e:
            print(e)


class OpMakeListOfVol(OpBase):
    def __init__(self, ser_inx_to_use: dict, subseq_to_use: dict, exp_patients: list, **kwargs):
        super().__init__(**kwargs)
        self._ser_inx_to_use = ser_inx_to_use
        self._subseq_to_use = subseq_to_use
        self._exp_patients = exp_patients

    def __call__(self, sample_dict: NDict, key_sequence_volumes_prefix: str, key_sequence_path_prefix: str,
                 key_sequence_series_num_prefix: str, key_sequence_ids: str):
        """
        extract_list_of_rel_vol extract the volume per seq based on SER_INX_TO_USE
        and put in one list
        :param vols_dict: dict of sitk vols per seq
        :param seq_info: dict of seq description per seq
        :return:
        """
        vols_dict = {k[len(key_sequence_volumes_prefix):]: sample_dict[k] for k in sample_dict if k.startswith(key_sequence_volumes_prefix)}
        img_path_dict = {k[len(key_sequence_path_prefix):]: sample_dict[k] for k in sample_dict if k.startswith(key_sequence_path_prefix)}

        def get_zeros_vol(vol):

            if vol.GetNumberOfComponentsPerPixel() > 1:
                ref_zeros_vol = sitk.VectorIndexSelectionCast(vol, 0)
            else:
                ref_zeros_vol = vol
            zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
            zeros_vol = sitk.GetImageFromArray(zeros_vol)
            zeros_vol.CopyInformation(ref_zeros_vol)
            return zeros_vol

        def stack_rel_vol_in_list(vols, series_inx_to_use, seq):
            # sequences with special fix
            B_SER_FIX = ['diffusie-3Scan-4bval_fs',
                         'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
                         'diff tra b 50 500 800 WIP511b alle spoelen']   #TODO: explain
            vols_list = []
            patient_id = sample_dict['data.sample_id'][0]
            use_series_index = patient_id in self._exp_patients  #todo: explain
            for s, v0 in vols.items():
                vol_inx_to_use = series_inx_to_use[patient_id][s] if use_series_index else series_inx_to_use['all'][s]

                if isinstance(vol_inx_to_use, list):
                    for inx in vol_inx_to_use:
                        if len(v0) == 0:
                            vols_list.append(get_zeros_vol(vols_list[0]))
                        elif len(v0) < inx + 1:
                            v = v0[len(v0) - 1]
                            vols_list.append(v)
                        else:
                            v = v0[inx]
                            if len(v):
                                vols_list.append(v)
                            else:
                                vols_list.append(get_zeros_vol(vols_list[0]))
                                if self._verbose:
                                    print('\n - problem with reading %s volume!' % s)


                else:
                    v = v0[vol_inx_to_use]
                    if len(v):
                        vols_list.append(v)
                    else:
                        vols_list.append(get_zeros_vol(vols_list[0]))
                        if self._verbose:
                            print('\n - problem with reading %s volume!' % s)

            if ('b' in img_path_dict):
                if (img_path_dict['b'][0] in B_SER_FIX):
                    vols_list[seq.index('b800')].CopyInformation(vols_list[seq.index('ADC')])
                    vols_list[seq.index('b400')].CopyInformation(vols_list[seq.index('ADC')])

            if len(vols_list) != len(seq):
                raise ValueError('Expected %d image modalities, found %d' % (len(seq), len(vols_list)))

            return vols_list

        # ------------------------
        # stack volumes by  seq order,
        # keep only vol as defined in series_inx_to_use
        if 'b_mix' in vols_dict:
            vols_dict.pop('b_mix') #todo: remove from sample_dictinary
            img_path_dict.pop('b_mix')

        # handle multiphase DCE in different series
        if (('DCE_mix_ph1' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph1']) > 0)) | \
                (('DCE_mix_ph2' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph2']) > 0)) | \
                (('DCE_mix_ph3' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph3']) > 0)):

            new_seq_id = 'DCE_mix'  # todo: add assert that does not exist
            seq_ids_to_group = [seq_id for seq_id in list(vols_dict) if f'{new_seq_id}_' in seq_id]
            vols_dict[new_seq_id] = []
            img_path_dict[new_seq_id] = []
            img_path_key = f'{key_sequence_path_prefix}{new_seq_id}'
            sample_dict[img_path_key] = []
            for key in seq_ids_to_group:
                stk_vols = vols_dict[key]
                series_path = img_path_dict[key]
                vols_dict[new_seq_id] += stk_vols  #todo: append to sample_dict
                sample_dict[img_path_key].append(series_path)
                vols_dict.pop(key) #todo: remove from sample_dictinary
                img_path_dict.pop(key)

        if ('DCE_mix_ph' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph']) > 0):  #todo: can this if happen but not the former?
            new_seq_id = 'DCE_mix'
            seq_ids_to_group = [seq_id for seq_id in sample_dict[key_sequence_ids] if f'{new_seq_id}_' in seq_id]
            img_path_key = f'{key_sequence_path_prefix}{new_seq_id}'
            if img_path_key not in sample_dict:
                sample_dict[img_path_key] = []
                vols_dict[new_seq_id] = []
                img_path_dict[new_seq_id] = []
            for seq_id in seq_ids_to_group:
                stk_vols = sample_dict[f'{key_sequence_volumes_prefix}{seq_id}']
                series_path = img_path_dict[seq_id]
                if (len(stk_vols) > 0):
                    inx_sorted = np.argsort(sample_dict[f'{key_sequence_series_num_prefix}{seq_id}'])
                    for ser_num_inx in inx_sorted:
                        vols_dict[new_seq_id] += [stk_vols[int(ser_num_inx)]]
                        img_path_dict[new_seq_id].append(series_path)

                vols_dict.pop(key)  #todo: remove from sample_dictinary
                img_path_dict.pop(key)

        vols_list = stack_rel_vol_in_list(vols_dict, self._ser_inx_to_use, self._subseq_to_use)
        sample_dict['vols_list'] = vols_list
        sample_dict['seq_dir'] = img_path_dict
        return sample_dict


class OpResampleStkVolsBasedRef(OpBase):
    def __init__(self, reference_inx: int, interpolation: str, **kwargs):
        super().__init__(**kwargs)
        assert reference_inx is not None #todo: redundant??
        self.reference_inx = reference_inx
        self.interpolation = interpolation

    def __call__(self, sample_dict: NDict, key_seq_volumes_prefix:str, key_out_prefix:str, key_seq_ids: str):

        # ------------------------
        # create resampling operator based on ref vol
        seq_ids = sample_dict[key_seq_ids]
        ref_seq_id = seq_ids[self.reference_inx]
        ref_seq_volumes = sample_dict[f'{key_seq_volumes_prefix}{ref_seq_id}']
        assert len(ref_seq_volumes) == 1
        ref_seq_volume = sitk.Cast(ref_seq_volumes[0], sitk.sitkFloat32) #todo: verify


        resample = self.create_resample(ref_seq_volume, self.interpolation, size=ref_seq_volume.GetSize(),
                                        spacing=ref_seq_volume.GetSpacing())

        for i_seq, seq_id in enumerate(seq_ids):
            seq_volumes = sample_dict[f'{key_seq_volumes_prefix}{seq_id}']
            if i_seq == self.reference_inx:
                seq_volumes_resampled = [ref_seq_volume] # do nothing
            else:
                seq_volumes_resampled = [resample.Execute( sitk.Cast(vol, sitk.sitkFloat32)) for vol in seq_volumes]
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


class OpStackList4DStk(OpBase):
    def __init__(self, reference_inx: int, **kwargs):
        super().__init__(**kwargs)
        self._reference_inx = reference_inx

    def __call__(self, sample_dict: NDict,
                 vols_stk_list: list,
                 ):
        vol_arr = [sitk.GetArrayFromImage(vol) for vol in (vols_stk_list)]
        vol_final = np.stack(vol_arr, axis=-1)
        vol_final_sitk = sitk.GetImageFromArray(vol_final, isVector=True)
        vol_final_sitk.CopyInformation(vols_stk_list[self._reference_inx])
        sample_dict['vol4D'] = vol_final_sitk
        return sample_dict


class OpRescale4DStk(OpBase):
    def __init__(self, mask_ch_inx: int, **kwargs):
        super().__init__(**kwargs)
        self._mask_ch_inx = mask_ch_inx

    def __call__(self, sample_dict: NDict,
                 stk_vol_4D: list,
                 ):

        # ------------------------
        # rescale intensity

        vol_backup = sitk.Image(stk_vol_4D)
        vol_array = sitk.GetArrayFromImage(stk_vol_4D)
        if len(vol_array.shape) < 4:
            vol_array = vol_array[:, :, :, np.newaxis]
        vol_array_pre_rescale = vol_array.copy()
        vol_array = self.apply_rescaling(vol_array)

        if self._mask_ch_inx:
            bool_mask = np.zeros(vol_array_pre_rescale[:, :, :, self._mask_ch_inx].shape)
            bool_mask[vol_array_pre_rescale[:, :, :, self._mask_ch_inx] > 0.3] = 1
            vol_array[:, :, :, self._mask_ch_inx] = bool_mask

        vol_final = sitk.GetImageFromArray(vol_array, isVector=True)
        vol_final.CopyInformation(vol_backup)
        vol_final = sitk.Image(vol_final)
        sample_dict['vol4D'] = vol_final
        return sample_dict

    def apply_rescaling(self, img: np.array, thres: tuple = (1.0, 99.0), method: str = 'noclip'):
        """
        apply_rescaling rescale each channal using method
        :param img:
        :param thres:
        :param method:
        :return:
        """
        eps = 0.000001

        def rescale_single_channel_image(img):
            # Deal with negative values first
            min_value = np.min(img)
            if min_value < 0:
                img -= min_value
            if method == 'clip':
                val_l, val_h = np.percentile(img, thres)
                img2 = img
                img2[img < val_l] = val_l
                img2[img > val_h] = val_h
                img2 = (img2.astype(np.float32) - val_l) / (val_h - val_l + eps)
            elif method == 'mean':
                img2 = img / max(np.mean(img), 1)
            elif method == 'median':
                img2 = img / max(np.median(img), 1)
                # write as op
                ######################
            elif method == 'noclip':
                val_l, val_h = np.percentile(img, thres)
                img2 = img
                img2 = (img2.astype(np.float32) - val_l) / (val_h - val_l + eps)
            else:
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


# -------------------------------------------------processor




if __name__ == "__main__":
    main()
