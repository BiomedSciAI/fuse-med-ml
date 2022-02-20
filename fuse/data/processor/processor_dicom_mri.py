"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""
import logging
import os, glob
import numpy as np
import SimpleITK as sitk
import pydicom
from scipy.ndimage.morphology import binary_dilation
import logging
import h5py
from typing import Tuple
import pandas as pd
from fuse.data.processor.processor_base import FuseProcessorBase

# ========================================================================
# sequences to be read, and the sequence name
SEQ_DICT = \
    {
        't2_tse_tra': 'T2',
        't2_tse_tra_Grappa3': 'T2',
        't2_tse_tra_320_p2': 'T2',

        'ep2d-advdiff-3Scan-high bvalue 100': 'b',
        'ep2d-advdiff-3Scan-high bvalue 500': 'b',
        'ep2d-advdiff-3Scan-high bvalue 1400': 'b',
        'ep2d_diff_tra2x2_Noise0_FS_DYNDISTCALC_BVAL': 'b',

        'ep2d_diff_tra_DYNDIST': 'b_mix',
        'ep2d_diff_tra_DYNDIST_MIX': 'b_mix',
        'diffusie-3Scan-4bval_fs': 'b_mix',
        'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen': 'b_mix',
        'diff tra b 50 500 800 WIP511b alle spoelen': 'b_mix',

        'ep2d_diff_tra_DYNDIST_MIX_ADC': 'ADC',
        'diffusie-3Scan-4bval_fs_ADC': 'ADC',
        'ep2d-advdiff-MDDW-12dir_spair_511b_ADC': 'ADC',
        'ep2d-advdiff-3Scan-4bval_spair_511b_ADC': 'ADC',
        'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC': 'ADC',
        'diff tra b 50 500 800 WIP511b alle spoelen_ADC': 'ADC',
        'ADC_S3_1': 'ADC',
        'ep2d_diff_tra_DYNDIST_ADC': 'ADC',

    }

# patients with special fix
EXP_PATIENTS = ['ProstateX-0191', 'ProstateX-0148', 'ProstateX-0180']

SEQ_TO_USE = ['T2', 'b', 'b_mix', 'ADC', 'ktrans']
SUB_SEQ_TO_USE = ['T2', 'b400', 'b800', 'ADC', 'ktrans']
SER_INX_TO_USE = {}
SER_INX_TO_USE['all'] = {'T2': -1, 'b': [0, 2], 'ADC': 0, 'ktrans': 0}
SER_INX_TO_USE['ProstateX-0148'] = {'T2': 1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}
SER_INX_TO_USE['ProstateX-0191'] = {'T2': -1, 'b': [0, 0], 'ADC': 0, 'ktrans': 0}
SER_INX_TO_USE['ProstateX-0180'] = {'T2': -1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}

# sequences with special fix
B_SER_FIX = ['diffusie-3Scan-4bval_fs',
             'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
             'diff tra b 50 500 800 WIP511b alle spoelen']

class FuseDicomMRIProcessor(FuseProcessorBase):
    def __init__(self,verbose: bool=True,reference_inx: int=0,seq_dict:dict=SEQ_DICT,
                 seq_to_use:list=SEQ_TO_USE,subseq_to_use:list=SUB_SEQ_TO_USE,
                 ser_inx_to_use:dict=SER_INX_TO_USE,exp_patients:dict=EXP_PATIENTS,
                 use_order_indicator: bool=False):
        '''
        FuseDicomMRIProcessor is MRI volume processor
        :param verbose: if print verbose
        :param reference_inx: index for the sequence that is selected as reference from SEQ_TO_USE (0 for T2)
        :param seq_dict: dictionary in which varies series descriptions are grouped
        together based on dict key.
        :param seq_to_use: The sequences to use are selected
        :param subseq_to_use:
        :param ser_inx_to_use: The series index to use
        :param exp_patients: patients with missing series that are treated in a special inx
        default params are for prostate_x dataset
        '''

        self._verbose = verbose
        self._reference_inx = reference_inx
        self._seq_dict = seq_dict
        self._seq_to_use = seq_to_use
        self._subseq_to_use = subseq_to_use
        self._ser_inx_to_use = ser_inx_to_use
        self._exp_patients = exp_patients
        self._use_order_indicator = use_order_indicator





    def __call__(self,
                 sample_desc,
                 *args, **kwargs):
        """
        sample_desc contains:
        :param images_path: path to directory in which dicom data is located
        :param ktrans_data_path: path to directory of Ktrans seq (prostate x)
        :param patient_id: patient indicator
        :return: 4D tensor of MRI volumes, reference volume
        """

        imgs_path, ktrans_data_path, patient_id = sample_desc

        self._imgs_path = imgs_path
        self._ktrans_data_path = ktrans_data_path
        self._patient_id = patient_id


        # ========================================================================
        # extract stk vol list per sequence
        vols_dict, seq_info = self.extract_vol_per_seq()

        # ========================================================================
        # list of sitk volumes (z,x,y) per sequence
        # order of volumes as defined in SER_INX_TO_USE
        # if missing volume, replaces with volume of zeros
        vol_list = self.extract_list_of_rel_vol(vols_dict, seq_info)
        vol_ref = vol_list[self._reference_inx]
        # ========================================================================
        # vol_4D is multichannel volume (z,x,y,chan(sequence))
        vol_4D = self.preprocess_and_stack_seq(vol_list, reference_inx=self._reference_inx)

        return vol_4D,vol_ref

    # ========================================================================

    def extract_stk_vol(self,img_path:str, img_list:list=[str], reverse_order:bool=False, is_path:bool=True)->list:
        """
        extract_stk_vol loads dicoms into sitk vol
        :param img_path: path to dicoms - load all dicoms from this path
        :param img_list: list of dicoms to load
        :param reverse_order: sometimes reverse dicoms orders is needed
        (for b series in which more than one sequence is provided inside the img_path)
        :param is_path: if True loads all dicoms from img_path
        :return: list of stk vols
        """

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
                return stk_vols

            elif is_path:
                vol = sitk.ReadImage(img_path)
                stk_vols.append(vol)
                return stk_vols

            else:
                series_reader = sitk.ImageSeriesReader()

                if img_list == []:
                    img_list = [series_reader.GetGDCMSeriesFileNames(img_path)]

                for n, imgs_names in enumerate(img_list):
                    if img_path not in img_list[0][0]:
                        imgs_names = [os.path.join(img_path, n) for n in imgs_names]
                    dicom_names = imgs_names[::-1] if reverse_order else imgs_names
                    series_reader.SetFileNames(dicom_names)
                    imgs = series_reader.Execute()
                    stk_vols.append(imgs)

                return stk_vols

        except Exception as e:
                print(e)







    # ========================================================================

    def sort_dicom_by_dicom_field(self,dcm_files: list, dicom_field: tuple =(0x19, 0x100c))->list:
        """
        sort_dicom_by_dicom_field sorts the dcm_files based on dicom_field
        For some MRI sequences different kinds of MRI series are mixed together (as in bWI) case
        This function creates a dict={dicom_field_type:list of relevant dicoms},
        than concats all to a list of the different series types

        :param dcm_files: list of all dicoms , mixed
        :param dicom_field: dicom field to sort based on
        :return: sorted_names_list, list of sorted dicom series
        """

        dcm_values = {}
        dcm_patient_z = {}
        dcm_instance = {}
        for index,dcm in enumerate(dcm_files):
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
                #sort by
                if index==0:
                    patient_z_ = []
                    for dcm_ in dcm_files:
                        dcm_ds_ = pydicom.dcmread(dcm_)
                        patient_z_.append(dcm_ds_.ImagePositionPatient[2])
                val = int(np.floor((instance_num-1)/len(np.unique(patient_z_))))
                if val not in dcm_values:
                    dcm_values[val] = []
                    dcm_patient_z[val] =[]
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
            sorted_names_list2 = [list(np.array(list_of_names)[np.argsort(list_of_z)]) for list_of_names,list_of_z in zip(sorted_names_list,dcm_patient_z_list)]
        else:
            # sort by instance number
            sorted_names_list2 = [list(np.array(list_of_names)[np.argsort(list_of_z)]) for list_of_names,list_of_z in zip(sorted_names_list,dcm_instance_list)]

        return sorted_names_list2


    # ========================================================================

    def extract_vol_per_seq(self)-> dict:
        """
        extract_vol_per_seq arranges sequences in sitk volumes dict
        dict{seq_description: list of sitk}
        :return:
        vols_dict, dict{seq_description: list of sitk}
        sequences_dict,dict{seq_description: list of series descriptions}
        """

        ktrans_path = os.path.join(self._ktrans_data_path, self._patient_id)

        if self._verbose:
            print('Patient ID: %s' % (self._patient_id))

        # ------------------------
        # images dict and sequences description dict

        vols_dict = {k: [] for k in self._seq_to_use}
        sequences_dict = {k: [] for k in self._seq_to_use}
        sequences_num_dict = {k: [] for k in self._seq_to_use}

        for img_path in os.listdir(self._imgs_path):
            try:
                full_path = os.path.join(self._imgs_path, img_path)
                dcm_files = glob.glob(os.path.join(full_path, '*.dcm'))
                series_desc = pydicom.dcmread(dcm_files[0]).SeriesDescription
                try:
                    series_num = int(pydicom.dcmread(dcm_files[0]).AcquisitionNumber)
                except:
                    series_num = int(pydicom.dcmread(dcm_files[0]).SeriesNumber)


                #------------------------
                # print series description
                series_desc_general = self._seq_dict[series_desc] \
                    if series_desc in self._seq_dict else 'UNKNOWN'
                if self._verbose:
                    print('\t- Series description:',' %s (%s)' % (series_desc, series_desc_general))



                #------------------------
                # ignore UNKNOWN series
                if series_desc not in self._seq_dict or \
                        self._seq_dict[series_desc] not in self._seq_to_use:
                    continue

                #------------------------
                # b-series - sorting images by b-value

                if self._seq_dict[series_desc] == 'b_mix':
                    dcm_ds = pydicom.dcmread(dcm_files[0])
                    if 'DiffusionBValue' in dcm_ds:
                        dicom_field = (0x0018,0x9087)#'DiffusionBValue'
                    else:
                        dicom_field = (0x19, 0x100c)

                    if self._use_order_indicator:
                        reverse_order = False
                    else:
                        #default
                        reverse_order = True

                    sorted_dicom_names = self.sort_dicom_by_dicom_field(dcm_files, dicom_field=dicom_field)
                    stk_vols = self.extract_stk_vol(full_path, img_list=sorted_dicom_names, reverse_order=reverse_order, is_path=False)

                # ------------------------
                # MASK
                elif self._seq_dict[series_desc] == 'MASK':
                    dicom_field = (0x0020, 0x0011)#series number

                    if self._use_order_indicator:
                        reverse_order = False
                    else:
                        # default
                        reverse_order = True

                    sorted_dicom_names = self.sort_dicom_by_dicom_field(dcm_files, dicom_field=dicom_field)
                    stk_vols = self.extract_stk_vol(full_path, img_list=sorted_dicom_names, reverse_order=reverse_order,
                                                    is_path=False)

                #------------------------
                # DCE - sorting images by time phases
                elif  'DCE' in self._seq_dict[series_desc]:
                    dcm_ds = pydicom.dcmread(dcm_files[0])
                    if 'TemporalPositionIdentifier' in dcm_ds:
                        dicom_field = (0x0020, 0x0100) #Temporal Position Identifier
                    elif 'TemporalPositionIndex' in dcm_ds:
                        dicom_field = (0x0020, 0x9128)
                    else:
                        dicom_field = (0x0020, 0x0012)#Acqusition Number

                    if self._use_order_indicator:
                        reverse_order = False
                    else:
                        #default
                        reverse_order = False
                    sorted_dicom_names = self.sort_dicom_by_dicom_field(dcm_files,dicom_field=dicom_field)
                    stk_vols = self.extract_stk_vol(full_path, img_list=sorted_dicom_names, reverse_order=False, is_path=False)


                #------------------------
                # general case
                else:
                    # images are sorted based instance number
                    stk_vols = self.extract_stk_vol(full_path, img_list=[], reverse_order=False, is_path=False)

                #------------------------
                # volume dictionary

                if self._seq_dict[series_desc] == 'b_mix':
                    vols_dict['b'] += stk_vols
                    sequences_dict['b'] += [series_desc]
                    sequences_num_dict['b']+=[series_num]
                else:
                    vols_dict[self._seq_dict[series_desc]] += stk_vols
                    sequences_dict[self._seq_dict[series_desc]] += [series_desc]
                    sequences_num_dict[self._seq_dict[series_desc]] += [series_num]

            except Exception as e:
                print(e)

        #------------------------
        # Read ktrans image
        try:

            if  glob.glob(os.path.join(ktrans_path, '*.mhd')):
                mhd_path = glob.glob(os.path.join(ktrans_path, '*.mhd'))[0]
                print('\t- Reading: %s (%s) (%s)' % (os.path.split(mhd_path)[-1], 'Ktrans', 'ktrans'))
                stk_vols = self.extract_stk_vol(mhd_path, img_list=[], reverse_order=False, is_path=True)
                vols_dict['ktrans'] = stk_vols
                sequences_dict['ktrans'] = [ktrans_path]


        except Exception as e:
            print(e)

        if 'b_mix' in vols_dict.keys():
            vols_dict.pop('b_mix')
            sequences_dict.pop('b_mix')

        # handle multiphase DCE in different series
        if ('DCE_mix_ph1' in vols_dict.keys()) | ('DCE_mix_ph2' in vols_dict.keys()) | ('DCE_mix_ph3' in vols_dict.keys()):
            if (len(vols_dict['DCE_mix_ph1'])>0) | (len(vols_dict['DCE_mix_ph2'])>0) | (len(vols_dict['DCE_mix_ph3'])>0):
                keys_list = [tmp for tmp in  list(vols_dict.keys()) if 'DCE_mix_' in tmp]
                for key in keys_list:
                    stk_vols = vols_dict[key]
                    series_desc = sequences_dict[key]
                    vols_dict['DCE_mix'] += stk_vols
                    sequences_dict['DCE_mix'] += [series_desc]
                    vols_dict.pop(key)
                    sequences_dict.pop(key)

        if ('DCE_mix_ph' in vols_dict.keys()):
            if (len(vols_dict['DCE_mix_ph'])>0):
                keys_list = [tmp for tmp in  list(sequences_num_dict.keys()) if 'DCE_mix_' in tmp]
                for key in keys_list:
                    stk_vols = vols_dict[key]
                    if (len(stk_vols)>0):
                        inx_sorted = np.argsort(sequences_num_dict[key])
                        for ser_num_inx in inx_sorted:
                            vols_dict['DCE_mix'] += [stk_vols[int(ser_num_inx)]]
                            sequences_dict['DCE_mix'] += [series_desc]
                    vols_dict.pop(key)
                    sequences_dict.pop(key)
        return vols_dict, sequences_dict

    # ========================================================================
    def extract_list_of_rel_vol(self,vols_dict:dict,seq_info:dict)->list:
        """
        extract_list_of_rel_vol extract the volume per seq based on SER_INX_TO_USE
        and put in one list
        :param vols_dict: dict of sitk vols per seq
        :param seq_info: dict of seq description per seq
        :return:
        """

        def get_zeros_vol(vol):

            if vol.GetNumberOfComponentsPerPixel() > 1:
                ref_zeros_vol = sitk.VectorIndexSelectionCast(vol, 0)
            else:
                ref_zeros_vol = vol
            zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
            zeros_vol = sitk.GetImageFromArray(zeros_vol)
            zeros_vol.CopyInformation(ref_zeros_vol)
            return zeros_vol

        def stack_rel_vol_in_list(vols,series_inx_to_use,seq):
            vols_list = []
            for s, v0 in vols.items():
                vol_inx_to_use = series_inx_to_use['all'][s]

                if self._patient_id in self._exp_patients:
                    vol_inx_to_use = series_inx_to_use[self._patient_id][s]

                if isinstance(vol_inx_to_use,list):
                    for inx in vol_inx_to_use:
                        if len(v0)==0:
                            vols_list.append(get_zeros_vol(vols_list[0]))
                        elif len(v0)<inx+1:
                            v = v0[len(v0)-1]
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


            if ('b' in seq_info.keys()):
                if (seq_info['b'][0] in B_SER_FIX):
                    vols_list[seq.index('b800')].CopyInformation(vols_list[seq.index('ADC')])
                    vols_list[seq.index('b400')].CopyInformation(vols_list[seq.index('ADC')])

            if len(vols_list) != len(seq):
                raise ValueError('Expected %d image modalities, found %d' % (len(seq), len(vols_list)))

            return vols_list


        # ------------------------
        # stack volumes by  seq order,
        # keep only vol as defined in series_inx_to_use
        vols_list = stack_rel_vol_in_list(vols_dict, self._ser_inx_to_use,self._subseq_to_use)

        return vols_list

    # ========================================================================
    def preprocess_and_stack_seq(self,vols:list,interpolation:str='bspline', reference_inx:int=0, mask_ch_inx:int=-1):

        """
        preprocess_and_stack_seq apply 4 preprocessing actions:
        1) cast vols to float32
        2) resample all voluemes based on ref vol
        3) stack all vol from vols list in one 4D tensor
        4) rescale vol
        :param vols:
        :param interpolation:
        :param reference_inx:
        :return:
        list of sitk vols
        """
        # ------------------------
        # casting to float32
        vols = [sitk.Cast(im, sitk.sitkFloat32) for im in vols]

        # ------------------------
        # create resampling oparator based on ref vol

        # define the reference volume
        vol_ref = vols[reference_inx]
        other_inx = list(set(range(0, len(vols))) - set([reference_inx]))

        resample = self.create_resample(vol_ref, interpolation,size=vol_ref.GetSize() ,spacing=vol_ref.GetSpacing())

        vols_res = []
        for i,vol in enumerate(vols):
            if i in other_inx:
                vol_res = resample.Execute(vol)
            else:
                vol_res = vol_ref
            vols_res.append(vol_res)

        # ------------------------
        # stack sequences in 4D sitk

        vol_arr = [sitk.GetArrayFromImage(vol) for vol in (vols_res)]
        vol_final = np.stack(vol_arr, axis=-1)
        vol_final_sitk = sitk.GetImageFromArray(vol_final, isVector=True)
        vol_final_sitk.CopyInformation(vol_ref)

        # ------------------------
        # rescale intensity

        vol_backup = sitk.Image(vol_final_sitk)
        vol_array = sitk.GetArrayFromImage(vol_final_sitk)
        if len(vol_array.shape)<4:
            vol_array= vol_array[:,:,:,np.newaxis]
        vol_array_pre_rescale = vol_array.copy()
        vol_array = self.apply_rescaling(vol_array)

        if mask_ch_inx:
            bool_mask = np.zeros(vol_array_pre_rescale[:,:,:,mask_ch_inx].shape)
            bool_mask[vol_array_pre_rescale[:,:,:,mask_ch_inx]>0.3] = 1
            vol_array[:,:,:,mask_ch_inx] = bool_mask

        vol_final = sitk.GetImageFromArray(vol_array, isVector=True)
        vol_final.CopyInformation(vol_backup)
        vol_final = sitk.Image(vol_final)

        return vol_final

    # ========================================================================
    def apply_rescaling(self,img:np.array, thres:tuple=(1.0, 99.0), method:str='noclip'):
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

    # ========================================================================
    def create_resample(self,vol_ref:sitk.sitkFloat32, interpolation: str, size:Tuple[int,int,int], spacing: Tuple[float,float,float]):
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



