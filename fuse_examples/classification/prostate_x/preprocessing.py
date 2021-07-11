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

import os, glob
import numpy as np
import SimpleITK as sitk
import pydicom
from scipy.ndimage.morphology import binary_dilation


# ========================================================================
# sequences to be read, and the sequence name
SEQ_DICT_PROSTATEX = \
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

SEQ_TO_USE = ['T2', 'b', 'b_mix', 'ADC', 'ktrans']
SUB_SEQ_TO_USE = ['T2', 'b400', 'b800', 'ADC', 'ktrans']

# sequences with special fix
B_SER_FIX = ['diffusie-3Scan-4bval_fs',
             'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
             'diff tra b 50 500 800 WIP511b alle spoelen']

# patients with special fix
EXP_PATIENTS = ['ProstateX-0191', 'ProstateX-0148', 'ProstateX-0180']

# use T2[-1],b400 [1],b800 [2]
SER_INX_TO_USE = {}
SER_INX_TO_USE['all'] = {'T2': -1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}
SER_INX_TO_USE['ProstateX-0148'] = {'T2': 1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}
SER_INX_TO_USE['ProstateX-0191'] = {'T2': -1, 'b': [0, 0], 'ADC': 0, 'ktrans': 0}
SER_INX_TO_USE['ProstateX-0180'] = {'T2': -1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}

# ==========================================================================
def create_resample(vol_ref, interpolation,size,spacing):
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

class PROSTATEX3DVolume:
    def __init__(self,
                 patient_id: str=None,
                 imgs_path: str=None,
                 ktrans_data_path: str=None,
                 array: np.ndarray = None,
                 hdf5_filename: str = None,
                 verbose: bool = True,
                 ):

        self._patient_id = patient_id
        self._imgs_path = imgs_path
        self._ktrans_data_path = ktrans_data_path
        self._array = array
        self._hdf5_filename = hdf5_filename
        self._verbose = verbose
        if array is None and imgs_path is None and hdf5_filename is None:
            print('ERROR link to volume OR path to images must be provided')
            assert False, 'ERROR while creating CAP3DVolume: no input was provided'

    # ========================================================================

    def extract_stk_vol(self,img_path, img_list=[], reverse_order=False, is_path=True):

            stk_vols = []
            if is_path:
                vol = sitk.ReadImage(img_path)
                stk_vols.append(vol)
                return stk_vols

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

    # ========================================================================

    def sort_dicom_by_dicom_field(self,dcm_files, dicom_field=(0x19, 0x100c)):
        dcm_values = {}
        for dcm in dcm_files:
            dcm_ds = pydicom.dcmread(dcm)

            val = int(dcm_ds[dicom_field].value)
            if val not in dcm_values:
                dcm_values[val] = []
            dcm_values[val].append(os.path.split(dcm)[-1])
        sorted_names_list = [v for k, v in dcm_values.items()]
        return sorted_names_list

    # ========================================================================
    def extract_prostatex_stk_vol(self):


        ktrans_path = os.path.join(self._ktrans_data_path, self._patient_id)

        if self._verbose: print('Patient ID: %s' % (self._patient_id))

        # ------------------------
        # images dict and sequences description dict

        vols_dict = {k: [] for k in SEQ_TO_USE}
        sequences_dict = {k: [] for k in SEQ_TO_USE}


        for img_path in os.listdir(self._imgs_path):
            try:
                full_path = os.path.join(self._imgs_path, img_path)
                dcm_files = glob.glob(os.path.join(full_path, '*.dcm'))
                series_desc = pydicom.dcmread(dcm_files[0]).SeriesDescription

                #------------------------
                # print series description
                series_desc_general = SEQ_DICT_PROSTATEX[series_desc] \
                    if series_desc in SEQ_DICT_PROSTATEX else 'UNKNOWN'
                if self._verbose: print('\t- Series description:',' %s (%s)' % (series_desc, series_desc_general))

                #------------------------
                # ignore UNKNOWN series
                if series_desc not in SEQ_DICT_PROSTATEX or \
                        SEQ_DICT_PROSTATEX[series_desc] not in SEQ_TO_USE:
                    continue

                #------------------------
                # b-series - sorting images by b-value

                if SEQ_DICT_PROSTATEX[series_desc] == 'b_mix':

                    sorted_dicom_names = self.sort_dicom_by_dicom_field(dcm_files, dicom_field=(0x19, 0x100c))
                    stk_vols = self.extract_stk_vol(full_path, img_list=sorted_dicom_names, reverse_order=True, is_path=False)
                #------------------------
                # general case
                else:
                    stk_vols = self.extract_stk_vol(full_path, img_list=[], reverse_order=False, is_path=False)

                #------------------------
                # volume dictionary

                if SEQ_DICT_PROSTATEX[series_desc] == 'b_mix':
                    vols_dict['b'] += stk_vols
                    sequences_dict['b'] += [series_desc]
                else:
                    vols_dict[SEQ_DICT_PROSTATEX[series_desc]] += stk_vols
                    sequences_dict[SEQ_DICT_PROSTATEX[series_desc]] += [series_desc]

            except Exception as e:
                print(e)

        # Read ktrans image
        try:

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
        return vols_dict, sequences_dict

    # ========================================================================
    def read_prostatex_sequences_patient(self):

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

                if self._patient_id in EXP_PATIENTS:
                    vol_inx_to_use = series_inx_to_use[self._patient_id][s]

                if isinstance(vol_inx_to_use,list):
                    for inx in vol_inx_to_use:
                        v = v0[inx]
                        if len(v):
                            vols_list.append(v)
                        else:
                            vols_list.append(get_zeros_vol(vols_list[0]))
                            if self._verbose: print('\n - problem with reading %s volume!' % s)

                else:
                    v = v0[vol_inx_to_use]
                    if len(v):
                        vols_list.append(v)
                    else:
                        vols_list.append(get_zeros_vol(vols_list[0]))
                        if self._verbose: print('\n - problem with reading %s volume!' % s)

            if ('b' in seq_info.keys()):
                if (seq_info['b'][0] in B_SER_FIX):
                    vols_list[seq.index('b800')].CopyInformation(vols_list[seq.index('ADC')])
                    vols_list[seq.index('b400')].CopyInformation(vols_list[seq.index('ADC')])

            if len(vols_list) != len(seq):
                raise ValueError('Expected %d image modalities, found %d' % (len(seq), len(vols_list)))

            return vols_list


        # ------------------------
        # extract stk vol list per sequence
        vols_dict, seq_info = self.extract_prostatex_stk_vol()

        # ------------------------
        # stack volumes by  seq order,
        # keep only vol as defined in series_inx_to_use
        vols_list = stack_rel_vol_in_list(vols_dict, SER_INX_TO_USE,SUB_SEQ_TO_USE)

        return vols_list


    # ========================================================================
    def apply_rescaling(self,img, thres=(1.0, 99.0), method='noclip'):

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
        # img[img < 0] = 0
        # Process each channel independently
        if len(img.shape) == 4:
            for i in range(img.shape[-1]):
                img[..., i] = rescale_single_channel_image(img[..., i])
        else:
            img = rescale_single_channel_image(img)

        return img


    # ========================================================================
    def apply_resampling(self,img, mask, spacing=(0.5, 0.5, 3), size=(160, 160, 32),
                             transform=None, interpolation='bspline',
                             label_interpolator=sitk.sitkLabelGaussian,
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


    # ========================================================================
    def preprocess_sequences(self,vols,interpolation='bspline', reference_inx=0):

        # ------------------------
        # casting to float32
        vols = [sitk.Cast(im, sitk.sitkFloat32) for im in vols]

        # ------------------------
        # create resampling oparator based on ref vol

        # define the reference volume
        vol_ref = vols[reference_inx]
        other_inx = list(set(range(0, len(vols))) - set([reference_inx]))

        resample = create_resample(vol_ref, interpolation,size=vol_ref.GetSize() ,spacing=vol_ref.GetSpacing())

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
        # print('1'+str(len(vol_arr)))
        vol_final = np.stack(vol_arr, axis=-1)
        # print('2'+str(vol_final.shape))
        vol_final_sitk = sitk.GetImageFromArray(vol_final, isVector=True)
        # print('3'+str(vol_final_sitk.GetSize()))
        vol_final_sitk.CopyInformation(vol_ref)

        # ------------------------
        # rescale intensity

        vol_backup = sitk.Image(vol_final_sitk)
        vol_array = sitk.GetArrayFromImage(vol_final_sitk)
        if len(vol_array.shape)<4:
            vol_array= vol_array[:,:,:,np.newaxis]
        # print('4' + str(vol_array.shape))
        vol_array = self.apply_rescaling(vol_array)
        vol_final = sitk.GetImageFromArray(vol_array, isVector=True)
        # print('5' + str(vol_final.GetSize()))
        vol_final.CopyInformation(vol_backup)
        vol_final = sitk.Image(vol_final)

        return vol_final


    # ========================================================================
    def crop_lesion_vol(self,vol, position, ref, size=(160, 160, 32), spacing=(1, 1, 3), center_slice=None):

        def get_lesion_mask(position, ref):
            mask = np.zeros_like(sitk.GetArrayViewFromImage(ref), dtype=np.uint8)

            coords = np.round(position[::-1]).astype(np.int)
            mask[coords[0], coords[1], coords[2]] = 1
            mask = binary_dilation(mask, np.ones((3, 5, 5))) + 0
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(ref)

            return mask_sitk

        mask = get_lesion_mask(position, ref)

        vol.SetOrigin((0,) * 3)
        mask.SetOrigin((0,) * 3)
        vol.SetDirection(np.eye(3).flatten())
        mask.SetDirection(np.eye(3).flatten())

        ma_centroid = mask > 0.5
        label_analysis_filer = sitk.LabelShapeStatisticsImageFilter()
        label_analysis_filer.Execute(ma_centroid)
        centroid = label_analysis_filer.GetCentroid(1)
        offset_correction = np.array(size) * np.array(spacing)/2
        corrected_centroid = np.array(centroid)
        corrected_centroid[2] = center_slice * np.array(spacing[2])
        offset = corrected_centroid - np.array(offset_correction)

        translation = sitk.TranslationTransform(3, offset)
        img, mask = self.apply_resampling(vol, mask, spacing=spacing, size=size, transform=translation)

        return img, mask


if __name__ == "__main__":
    path_to_db = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/virtual_biopsy/prostate/experiments/V1/'
    dataset = 'prostate_x'
    if dataset=='tcia':
        # for TCIA-Prostate-MRI-US-Biopsy
        prostate_data_path = '/projects/msieve/MedicalSieve/PatientData/TCIA-Prostate-MRI-US-Biopsy/manifest-1599764098812/'
        masks_data_path = prostate_data_path + '/STLs/'
        sample = ('18042021', 'train', 'Prostate-MRI-US-Biopsy-0007', 'pred')

        a = PROSTATEX3DVolume(path_to_db,prostate_data_path,masks_data_path,dataset,(16, 120, 120))
        samples = a.__call__(sample)
        a = 1
    elif dataset=='prostate_x':
        EXP_LESION_INX_PATIENTS = ['ProstateX-0005', 'ProstateX-0105', 'ProstateX-0154']
        EXP_LESION_REMOVE_PATIENTS = ['ProstateX-0025']

    # for ProstateX

        path_imgs = '/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044//PROSTATEx/ProstateX-0148/03-12-2012-MR prostaat kanker detectie WDSmc MCAPRODETW-33123'
        patient_id = 'ProstateX-0148'
        path_ktrans = '/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044//ProstateXKtrains-train-fixed/'

        vol3D = PROSTATEX3DVolume(patient_id=patient_id,imgs_path=path_imgs,ktrans_data_path=path_ktrans)

        reference_inx = 0
        lsn_shape = (12, 32, 32)
        spacing = (0.5, 0.5, 3)

        vol_list = vol3D.read_prostatex_sequences_patient()
        vol = vol3D.preprocess_sequences(vol_list, reference_inx=reference_inx)

        positions_vol = np.array([146.84003826 ,207.70670297,   4.00001143]) #T2 coord
        vol_cropped, mask_cropped = vol3D.crop_lesion_vol(vol, positions_vol, vol_list[reference_inx],
                                                          center_slice=positions_vol[2],
                                                          size=(lsn_shape[2], lsn_shape[1], lsn_shape[0]), spacing=spacing)
