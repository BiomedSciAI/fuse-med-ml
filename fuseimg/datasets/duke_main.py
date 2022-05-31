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
from fuseimg.datasets.duke import OpDukeSampleIDDecode, Duke, _get_sequence_2_series_desc_mapping
from tempfile import mkdtemp



def main2(output_file):
    cache_dir =  '/tmp/duke_cache_ps4y2jk'#mkdtemp(prefix="duke_cache")
    duke_dataset = Duke.dataset(sample_ids = ['Breast_MRI_900'],
                                cache_dir=cache_dir,
                                num_workers=0)
    print(duke_dataset[0])

# def main(output_file):
#     root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
#     data_path = os.path.join(root_path, 'Duke-Breast-Cancer-MRI')
#     metadata_path = os.path.join(root_path, 'metadata.csv')
#
#     SEQ_IDS = ['DCE_mix_ph1', 'DCE_mix_ph3']
#
#     def get_selected_series_index(sample_id, seq_id):
#         patient_id = sample_id[0]
#         if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
#             map = {'DCE_mix': [2], 'MASK': [0]}
#         else:
#             map = {'DCE_mix': [1], 'MASK': [0]}
#         return map[seq_id]
#
#     seq_dict = _get_sequence_2_series_desc_mapping(metadata_path)
#
#     TEST_PATIENT_ID, TEST_STUDY_ID = 'Breast_MRI_900', '01-01-1990-BREASTROUTINE DYNAMICS-51487'
#     sample_dict = NDict({'data':{'sample_id': (TEST_PATIENT_ID, TEST_STUDY_ID)}})
#     # step 1: map sample_ids to
#     op = OpDukeSampleIDDecode(data_path=data_path)
#     sample_dict = op(sample_dict=sample_dict, key_out='data.input.mri_path', op_id=1)
#
#     # step 2: read files info for the sequences
#     op = OpExtractDicomsPerSeq(seq_ids=SEQ_IDS, seq_dict=seq_dict, use_order_indicator=False)
#     sample_dict = op(sample_dict,
#                      key_in='data.input.mri_path',
#                      key_out_sequences='data.input.sequence_ids',
#                      key_out_path_prefix='data.input.path.',
#                      key_out_dicoms_prefix='data.input.dicoms.',
#                      key_out_series_num_prefix='data.input.series_num.'
#                      )
#
#     # step 3: Load STK volumes of MRI sequences
#     op = OpLoadDicomAsStkVol(reverse_order=False, is_file=False)
#     sample_dict = op(sample_dict,
#                      key_in_seq_ids='data.input.sequence_ids',
#                      key_in_path_prefix='data.input.path.',
#                      key_in_dicoms_prefix='data.input.dicoms.',
#                      key_out_prefix='data.input.volumes.')
#
#     # step 4: group DCE sequnces into DCE_mix
#     op = OpGroupDCESequences()
#     sample_dict = op(sample_dict,
#                      key_sequence_ids='data.input.sequence_ids',
#                      key_path_prefix='data.input.path.',
#                      key_series_num_prefix='data.input.series_num.',
#                      key_volumes_prefix='data.input.volumes.',
#                      )
#
#     # step 5: select single volume from DCE_mix sequence
#     op = OpSelectVolumes(subseq_to_use=['DCE_mix'], get_indexes_func=get_selected_series_index)
#     sample_dict = op(sample_dict,
#                      key_in_path_prefix='data.input.path.',
#                      key_in_volumes_prefix='data.input.volumes.',
#                      key_out_path_prefix='data.input.selected_path.',
#                      key_out_volumes_prefix='data.input.selected_volumes.')
#
#     op_resample_stk_vols_based_ref = OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline')
#     sample_dict = op_resample_stk_vols_based_ref(sample_dict,
#                                                  key_seq_ids='data.input.sequence_ids',
#                                                  key_seq_volumes_prefix='data.input.selected_volumes.',
#                                                  key_out_prefix='data.input.selected_volumes_resampled.',
#                                                  )
#     if output_file is not None:
#         save_pickle_safe(sample_dict, output_file, verbose=True)
#











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
    # main(output_file)
    main2(output_file)
    if output_file is not None:
        compare_sample_dicts(baseline_output_file, output_file)
