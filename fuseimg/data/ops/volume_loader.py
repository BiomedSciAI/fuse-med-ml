import os

print(os.environ["LD_LIBRARY_PATH"])
from fuse.data.ops.op_base import OpBase
from typing import Optional
import numpy as np
from fuse.data.ops.ops_common import OpApplyTypes
#import nibabel as nib
from fuse.utils.ndict import NDict
import pandas as pd
import pydicom
import glob
import h5py
import SimpleITK as sitk
from typing import Tuple

# put in dataset
def process_mri_series(metadata_path: str):

    seq_to_use = ['DCE_mix_ph1',
                  'DCE_mix_ph2',
                  'DCE_mix_ph3',
                  'DCE_mix_ph4',
                  'DCE_mix',
                  'DCE_mix_ph',
                  'MASK']
    subseq_to_use = ['DCE_mix_ph2', 'MASK']

    l_seq = pd.read_csv(metadata_path)
    seq_to_use_full = list(l_seq['Series Description'].value_counts().keys())

    SER_INX_TO_USE = {}
    SER_INX_TO_USE['all'] = {'DCE_mix': [1], 'MASK': [0]}
    SER_INX_TO_USE['Breast_MRI_120'] = {'DCE_mix': [2], 'MASK': [0]}
    SER_INX_TO_USE['Breast_MRI_596'] = {'DCE_mix': [2], 'MASK': [0]}
    exp_patients = ['Breast_MRI_120','Breast_MRI_596']
    opt_seq = [
        '1st','1ax','1Ax','1/ax',
        '2nd','2ax','2Ax','2/ax',
        '3rd','3ax','3Ax', '3/ax',
        '4th','4ax','4Ax','4/ax',
    ]
    my_keys = ['DCE_mix_ph1'] * 4 + ['DCE_mix_ph2'] * 4 + ['DCE_mix_ph3'] * 4 + ['DCE_mix_ph4'] * 4
    seq_to_use_full_slash = [s.replace('ax','/ax') for s in seq_to_use_full]
    seq_to_use_full_slash = [s.replace('Ax', '/Ax') for s in seq_to_use_full_slash]

    seq_to_use_dict = {}
    for opt_seq_tmp, my_key in zip(opt_seq, my_keys):
        if my_key not in seq_to_use_dict.keys():
            seq_to_use_dict[my_key] = []
        seq_to_use_dict[my_key] +=[s for s in seq_to_use_full if opt_seq_tmp in s]+[s for s in seq_to_use_full_slash if opt_seq_tmp in s]

    if 'DCE_mix_ph' not in seq_to_use_dict.keys():
        seq_to_use_dict['DCE_mix_ph'] = []
    seq_to_use_dict['DCE_mix_ph']+=['ax dyn']

    return seq_to_use_dict,SER_INX_TO_USE,exp_patients,seq_to_use,subseq_to_use

# sequences with special fix
B_SER_FIX = ['diffusie-3Scan-4bval_fs',
             'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
             'diff tra b 50 500 800 WIP511b alle spoelen']

#------------------------------------------------MRI processor

class OpExtractDicomsPerSeq(OpBase):
    '''
    Return location dir of requested sequence
    '''

    def __init__(self, dir_path, prepare_seq_dict_func, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path
        self._seq_dict = prepare_seq_dict_func

    '''
    :param key_in: the key name in sample_dict that holds the filename
    :param key_out:
    '''

    def __call__(self, sample_dict: NDict, key_in: (str, str), key_out: (str,str),seq_desc= str,seq_dict= dict, use_order_indicator: bool = False):
        self._use_order_indicator = use_order_indicator
        self.sample_dict = sample_dict

        patient_id = sample_dict[key_in[0]]
        study_id = sample_dict[key_in[1]]

        seq_dir, dicom_path = self.ExtractSeqDir(patient_id,study_id,seq_desc,seq_dict)
        series_num = self.ExtractSerNum(dicom_path)
        dicom_field = self.ExtractDicomField(dicom_path, seq_desc)
        sorted_dicom_list = self.SortDicomsByField(dicom_path,dicom_field)

        dicom_list = [os.path.join(dicom_path,dicom) for dicom in sorted_dicom_list[0]]
        if key_out[0] not in sample_dict.keys():
            sample_dict[key_out[0]] = {}
        if key_out[1] not in sample_dict.keys():
            sample_dict[key_out[1]] = {}
        if key_out[2] not in sample_dict.keys():
            sample_dict[key_out[2]] = {}

        sample_dict[key_out[0]][seq_desc] = seq_dir
        sample_dict[key_out[1]][seq_desc] = series_num
        sample_dict[key_out[2]][seq_desc] = dicom_list

        return sample_dict

    def ExtractSeqDir(self,patient_id,study_id,seq_desc,seq_dict):
        '''
        Return location dir of requested sequence
        '''
        '''
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out:
        '''
        img_filename = os.path.join(self._dir_path,patient_id,study_id)
        seq_list = os.listdir(img_filename)

        for seq in seq_dict[seq_desc]:
            match_seq = [y for y in seq_list if seq in y]
            if len(match_seq)>0:
                seq_dir= match_seq[0]
                dicom_path = os.path.join(data_path, patient_id, study_id,seq_dir)
                return seq_dir,dicom_path

        seq_dir = 'NAN'
        dicom_path = 'NAN'
        raise Exception(f"OpExtractSeqInfo: no {seq_desc} was found under path")
        return seq_dir, dicom_path

    def ExtractSerNum(self,dicom_path):

        '''
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out:
        '''

        dicom_filename = os.listdir(dicom_path)[0]
        dcm_ds = pydicom.dcmread(os.path.join(dicom_path,dicom_filename))

        # series number
        try:
            series_num = int(dcm_ds.AcquisitionNumber)
        except:
            series_num = int(dcm_ds.SeriesNumber)

        return series_num

    def ExtractDicomField(self,dicom_path,seq_desc):
        '''
        Return location dir of requested sequence
        '''

        '''
        :param key_in: the key name in sample_dict that holds the filename
        :param key_out:
        '''
        dicom_filename = os.listdir(dicom_path)[0]
        dcm_ds = pydicom.dcmread(os.path.join(dicom_path,dicom_filename))

        #dicom key
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
            dicom_field='NAN'

        return dicom_field

    def SortDicomsByField(self, dicom_path,dicom_field):
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
        dcm_files = glob.glob(os.path.join(dicom_path, '*.dcm'))
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

    def __init__(self, dir_path: str, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path

    def __call__(self, sample_dict: NDict,
                 img_path:str,
                 img_list:list,
                 reverse_order:bool,
                 is_path:bool,
                 seq_desc: str):
        """
        extract_stk_vol loads dicoms into sitk vol
        :param img_path: path to dicoms - load all dicoms from this path
        :param img_list: list of dicoms to load
        :param reverse_order: sometimes reverse dicoms orders is needed
        (for b series in which more than one sequence is provided inside the img_path)
        :param is_path: if True loads all dicoms from img_path
        :return: list of stk vols
        """
        if 'seq_vol' not in sample_dict.keys():
            sample_dict['seq_vol'] = {}

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

            elif is_path:
                vol = sitk.ReadImage(img_path)
                stk_vols.append(vol)

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

            sample_dict['seq_vol'][seq_desc] = stk_vols
            return sample_dict

        except Exception as e:
            print(e)

class OpMakeListOfVol(OpBase):
    def __init__(self, ser_inx_to_use: dict, subseq_to_use:dict, exp_patients:list, **kwargs):
        super().__init__(**kwargs)
        self._ser_inx_to_use = ser_inx_to_use
        self._subseq_to_use = subseq_to_use
        self._exp_patients = exp_patients

    def __call__(self, sample_dict: NDict,
                       vols_dict:dict,
                       seq_info:dict
                 ):
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

                if sample_dict['sample_id'] in self._exp_patients:
                    vol_inx_to_use = series_inx_to_use[sample_dict['sample_id']][s]

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
        if 'b_mix' in sample['seq_vol'].keys():
            vols_dict.pop('b_mix')
            seq_info.pop('b_mix')

        # handle multiphase DCE in different series
        if (('DCE_mix_ph1' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph1']) > 0)) | \
                (('DCE_mix_ph2' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph2']) > 0)) | \
                (('DCE_mix_ph3' in vols_dict.keys()) and (len(vols_dict['DCE_mix_ph3']) > 0)):

            keys_list = [tmp for tmp in list(vols_dict.keys()) if 'DCE_mix_' in tmp]
            vols_dict['DCE_mix'] = []
            seq_info['DCE_mix'] = []
            for key in keys_list:
                stk_vols = vols_dict[key]
                series_desc = seq_info[key]
                vols_dict['DCE_mix'] += stk_vols
                sample['seq_dir']['DCE_mix'] += [series_desc]
                vols_dict.pop(key)
                seq_info.pop(key)

        if ('DCE_mix_ph' in vols_dict.keys()):
            if (len(vols_dict['DCE_mix_ph']) > 0):
                keys_list = [tmp for tmp in list(sample['ser_num'].keys()) if 'DCE_mix_' in tmp]
                for key in keys_list:
                    stk_vols = sample['seq_vol'][key]
                    if (len(stk_vols) > 0):
                        inx_sorted = np.argsort(sample['ser_num'][key])
                        for ser_num_inx in inx_sorted:
                            vols_dict['DCE_mix'] += [stk_vols[int(ser_num_inx)]]
                            seq_info['DCE_mix'] += [series_desc]

                    vols_dict.pop(key)
                    seq_info.pop(key)

        vols_list = stack_rel_vol_in_list(vols_dict, self._ser_inx_to_use,self._subseq_to_use)
        sample_dict['vols_list'] = vols_list
        sample_dict['seq_dir'] = seq_info
        return sample_dict

class OpResampleStkVolsBasedRef(OpBase):
    def __init__(self, reference_inx: int, interpolation: str, **kwargs):
        super().__init__(**kwargs)
        self.reference_inx = reference_inx
        self.interpolation = interpolation

    def __call__(self, sample_dict: NDict,
                       vols: list
                 ):

        # ------------------------
        # casting to float32
        vols = [sitk.Cast(im, sitk.sitkFloat32) for im in vols]

        # ------------------------
        # create resampling operator based on ref vol

        if self.reference_inx is not None:
            # define the reference volume
            vol_ref = vols[self.reference_inx]
            other_inx = list(set(range(0, len(vols))) - set([self.reference_inx]))

            resample = self.create_resample(vol_ref, self.interpolation,size=vol_ref.GetSize() ,spacing=vol_ref.GetSpacing())

            vols_res = []
            for i,vol in enumerate(vols):
                if i in other_inx:
                    vol_res = resample.Execute(vol)
                else:
                    vol_res = vol_ref
                vols_res.append(vol_res)


        sample_dict['vols_res'] = vols_res
        return sample_dict


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

class OpStackList4DStk(OpBase):
    def __init__(self, reference_inx:int, **kwargs):
        super().__init__(**kwargs)
        self._reference_inx =  reference_inx

    def __call__(self, sample_dict: NDict,
                       vols_stk_list:list,
                 ):

        vol_arr = [sitk.GetArrayFromImage(vol) for vol in (vols_stk_list)]
        vol_final = np.stack(vol_arr, axis=-1)
        vol_final_sitk = sitk.GetImageFromArray(vol_final, isVector=True)
        vol_final_sitk.CopyInformation(vols_stk_list[self._reference_inx])
        sample_dict['vol4D']=vol_final_sitk
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

#-------------------------------------------------processor




if __name__ == "__main__":

    path_to_db = '.'
    root_data = '/gpfs/haifa/projects/m/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

    seq_dict,SER_INX_TO_USE,exp_patients,_,_ = process_mri_series(root_data+'/metadata.csv')

    sample={}
    sample['sample_id'] = 'Breast_MRI_900'
    sample['study_id'] = '01-01-1990-BREASTROUTINE DYNAMICS-51487'
    seq_desc = ['DCE_mix_ph1','DCE_mix_ph3']
    data_path = os.path.join(root_data, 'Duke-Breast-Cancer-MRI')



    # ------------------------------------------------MRI processor
    # Load volumes dict
    for seq_desc1 in seq_desc:
        op = OpExtractDicomsPerSeq(dir_path=data_path, #MICHAL: why create in loop and not outside?
                                   prepare_seq_dict_func=process_mri_series(root_data+'/metadata.csv'))
        sample = op.__call__(sample,
                             key_in = ('sample_id','study_id'),
                             key_out = ('seq_dir', 'seq_num','dicom_list'),
                             seq_desc = seq_desc1,
                             seq_dict= seq_dict,
                             use_order_indicator=False)

        op = OpLoadDicomAsStkVol(data_path)  #MICHAL: why create in loop and not outside?
        sample = op.__call__(sample,
                             img_path =sample['dicom_path'],img_list=sample['dicom_list'][seq_desc1],
                             reverse_order = False, is_path = False,seq_desc = seq_desc1)

        op = OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline')
        sample = op.__call__(sample, sample['vols_list'])

    ###########
    # get list

    op = OpMakeListOfVol(SER_INX_TO_USE, ['DCE_mix'],exp_patients)
    sample = op.__call__(sample, sample['seq_vol'], sample['seq_dir'])


    # op = OpStackListStk(reference_inx=0)
    # sample = op.__call__(sample,sample['vols_list'])
    #
    # op = OpRescale4DStk(mask_ch_inx=None)
    # sample = op.__call__(sample,sample['vol4D'])

    # -------------------------------------------------processor

    a=1



