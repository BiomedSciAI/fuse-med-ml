import os

from fuse.data.ops.op_base import OpBase
import numpy as np
# import nibabel as nib
from fuse.utils.ndict import NDict
import pandas as pd
import SimpleITK as sitk

from fuse.utils import NDict



from fuse.utils.file_io.file_io import load_pickle, save_pickle_safe
from fuseimg.data.ops.ops_mri import OpExtractDicomsPerSeq, OpLoadDicomAsStkVol, OpGroupDCESequences, OpSelectVolumes, OpResampleStkVolsBasedRef
from fuseimg.datasets.duke import OpDukeSampleIDDecode, Duke
from tempfile import mkdtemp



def main2(output_file):
    cache_dir = mkdtemp(prefix="duke_cache")
    duke_dataset = Duke.dataset(cache_dir=cache_dir)
    print("ok")

def main(output_file):
    root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
    data_path = os.path.join(root_path, 'Duke-Breast-Cancer-MRI')
    metadata_path = os.path.join(root_path, 'metadata.csv')

    SEQ_IDS = ['DCE_mix_ph1', 'DCE_mix_ph3']

    def get_selected_series_index(sample_id, seq_id):
        patient_id = sample_id[0]
        if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
            map = {'DCE_mix': [2], 'MASK': [0]}
        else:
            map = {'DCE_mix': [1], 'MASK': [0]}
        return map[seq_id]

    seq_dict = get_sequence_2_series_desc_mapping(metadata_path)

    TEST_PATIENT_ID, TEST_STUDY_ID = 'Breast_MRI_900', '01-01-1990-BREASTROUTINE DYNAMICS-51487'
    sample_dict = NDict({'data':{'sample_id': (TEST_PATIENT_ID, TEST_STUDY_ID)}})
    # step 1: map sample_ids to
    op = OpDukeSampleIDDecode(data_path=data_path)
    sample_dict = op(sample_dict=sample_dict, key_out='data.input.mri_path', op_id=1)

    # step 2: read files info for the sequences
    op = OpExtractDicomsPerSeq(seq_ids=SEQ_IDS, seq_dict=seq_dict, use_order_indicator=False)
    sample_dict = op(sample_dict,
                     key_in='data.input.mri_path',
                     key_out_sequences='data.input.sequence_ids',
                     key_out_path_prefix='data.input.path.',
                     key_out_dicoms_prefix='data.input.dicoms.',
                     key_out_series_num_prefix='data.input.series_num.'
                     )

    # step 3: Load STK volumes of MRI sequences
    op = OpLoadDicomAsStkVol(reverse_order=False, is_file=False)
    sample_dict = op(sample_dict,
                     key_in_seq_ids='data.input.sequence_ids',
                     key_in_path_prefix='data.input.path.',
                     key_in_dicoms_prefix='data.input.dicoms.',
                     key_out_prefix='data.input.volumes.')

    # step 4: group DCE sequnces into DCE_mix
    op = OpGroupDCESequences()
    sample_dict = op(sample_dict,
                     key_sequence_ids='data.input.sequence_ids',
                     key_path_prefix='data.input.path.',
                     key_series_num_prefix='data.input.series_num.',
                     key_volumes_prefix='data.input.volumes.',
                     )

    # step 5: select single volume from DCE_mix sequence
    op = OpSelectVolumes(subseq_to_use=['DCE_mix'], get_indexes_func=get_selected_series_index)
    sample_dict = op(sample_dict,
                     key_in_path_prefix='data.input.path.',
                     key_in_volumes_prefix='data.input.volumes.',
                     key_out_path_prefix='data.input.selected_path.',
                     key_out_volumes_prefix='data.input.selected_volumes.')

    op_resample_stk_vols_based_ref = OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline')
    sample_dict = op_resample_stk_vols_based_ref(sample_dict,
                                                 key_seq_ids='data.input.sequence_ids',
                                                 key_seq_volumes_prefix='data.input.selected_volumes.',
                                                 key_out_prefix='data.input.selected_volumes_resampled.',
                                                 )
    if output_file is not None:
        save_pickle_safe(sample_dict, output_file, verbose=True)


# put in dataset
def get_sequence_2_series_desc_mapping(metadata_path: str):
    # read metadata file and match between series_desc in metadata file and sequence
    metadata_df = pd.read_csv(metadata_path)
    series_description_list = metadata_df['Series Description'].unique()

    sequence_2_series_desc_mapping = {'DCE_mix_ph': ['ax dyn']}
    patterns = ['1st', '2nd', '3rd', '4th']
    for i_phase in range(1, 5):
        seq_id = f'DCE_mix_ph{i_phase}'
        sequence_2_series_desc_mapping[seq_id] = []
        phase_patterns = [patterns[i_phase - 1], f'{i_phase}ax', f'{i_phase}Ax', f'{i_phase}/ax', f'{i_phase}/Ax']

        for series_desc in series_description_list:
            has_match = any(p in series_desc for p in phase_patterns)
            if has_match:
                series_desc2 = series_desc.replace(f'{i_phase}ax', f'{i_phase}/ax').replace(f'{i_phase}Ax',
                                                                                            f'{i_phase}/Ax')
                sequence_2_series_desc_mapping[seq_id] += [series_desc, series_desc2]

    return sequence_2_series_desc_mapping


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




def compare_sample_dicts(file1, file2):
    d1 = load_pickle(file1).flatten()
    d2 = load_pickle(file2).flatten()
    print(len(d1), d1.keys())
    print(len(d2), d2.keys())
    # map = {'data.sample_id': 'data.sample_id', 'data.input.sequence_ids': 'data.input.sequence_ids',
    #  'data.input.sequence_path.DCE_mix': 'data.input.path.DCE_mix',
    #  'data.input.sequence_volumes.DCE_mix': 'data.input.volumes.DCE_mix',
    #  'data.input.sequence_selected_volume.DCE_mix': 'data.input.selected_volumes.DCE_mix',
    #  'data.input.sequence_selected_path.DCE_mix': 'data.input.selected_path.DCE_mix',
    #  'data.input.sequence_selected_volume_resampled.DCE_mix': 'data.input.selected_volumes_resampled.DCE_mix'}
    #
    # for s1, s2 in map.items():
    #     print(s1, s2, s2 in d2.keys(), d1[s1]==d2[s2])

    print(d1 == d2)


if __name__ == "__main__":
    baseline_output_file = '/user/ozery/output/baseline1.pkl'  # '/tmp/f2.pkl'
    output_file = None #'/user/ozery/output/f4.pkl'
    main(output_file)
    # main2(output_file)
    if output_file is not None:
        compare_sample_dicts(baseline_output_file, output_file)
