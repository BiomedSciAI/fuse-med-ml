from functools import partial
import os
from typing import Hashable, List, Optional, Sequence, Tuple, Union
import glob

import numpy as np
import pandas as pd
import pickle


from fuse.utils import NDict


from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault
from fuse.data.ops.op_base import OpBase, OpReversibleBase


from fuse.data.utils.sample import get_sample_id


from fuseimg.data.ops.ops_mri import OpExtractDicomsPerSeq, OpLoadDicomAsStkVol, OpGroupDCESequences, OpSelectVolumes, \
    OpResampleStkVolsBasedRef, OpStackList4DStk, OpRescale4DStk, OpCreatePatch


class OpDukeSampleIDDecode(OpReversibleBase):
    '''
    decodes sample id into path of MRI images
    '''

    def __init__(self, data_path:str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path

    def __call__(self, sample_dict:NDict, key_out:str, op_id: Optional[str]) -> NDict:

        sid = get_sample_id(sample_dict)

        sample_path_pattern = os.path.join(self._data_path, sid, '*')
        sample_path = glob.glob(sample_path_pattern)
        assert len(sample_path)==1
        sample_dict[key_out] = sample_path[0]

        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict


def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
        map = {'DCE_mix': [2], 'MASK': [0]}
    else:
        map = {'DCE_mix': [1], 'MASK': [0]}
    return map[seq_id]




class Duke:
    DUKE_DATASET_VER = 0
    @staticmethod
    def sample_ids():
        if False:
            # TEST_PATIENT_ID, TEST_STUDY_ID = 'Breast_MRI_900', '01-01-1990-BREASTROUTINE DYNAMICS-51487'
            TEST_PATIET_ID = 'Breast_MRI_900'

            test_sample_id = TEST_PATIENT_ID
            return [test_sample_id]
        return [f'Breast_MRI_{i:03d}' for i in range(1,923)]

    @staticmethod
    def static_pipeline(root_path, select_series_func) -> PipelineDefault:

        data_path = os.path.join(root_path, 'Duke-Breast-Cancer-MRI')
        metadata_path = os.path.join(root_path, 'metadata.csv')
        annotations_path = os.path.join('/user/ozery/msieve_dev3_ozery/workspace/whi/fuse-med-ml1/fuse_examples/classification/duke_breast_cancer/dataset_DUKE_folds_ver11102021TumorSize_seed1.pickle')
        with open(annotations_path, 'rb') as infile:
            fold_annotations_dict= pickle.load(infile)
            annotations_df = pd.concat([fold_annotations_dict[f'data_fold{fold}'] for fold in range(len(fold_annotations_dict))])

        def get_annotations(sample_id):
            patient_annotations_df = annotations_df[annotations_df['Patient ID'] == sample_id]
            return patient_annotations_df
        series_desc_2_sequence_map = _get_sequence_2_series_desc_mapping(metadata_path)
        seq_ids = ['DCE_mix_ph1',
                  'DCE_mix_ph2',
                  'DCE_mix_ph3',
                  'DCE_mix_ph4',
                  'DCE_mix',
                  'DCE_mix_ph',
                  'MASK']

        static_pipeline = PipelineDefault("static", [
            # step 1: map sample_ids to
            (OpDukeSampleIDDecode(data_path=data_path),
                    dict(key_out='data.input.mri_path')),
            # step 2: read files info for the sequences
            (OpExtractDicomsPerSeq(seq_ids=seq_ids, series_desc_2_sequence_map=series_desc_2_sequence_map, use_order_indicator=False),
                    dict(key_in='data.input.mri_path',
                                key_out_sequences='data.input.sequence_ids',
                                key_out_path_prefix='data.input.path.',
                                key_out_dicoms_prefix='data.input.dicoms.',
                                key_out_series_num_prefix='data.input.series_num.')
             ),
            # step 3: Load STK volumes of MRI sequences
            (OpLoadDicomAsStkVol(reverse_order=False, is_file=False),
                    dict(key_in_seq_ids='data.input.sequence_ids',
                         key_in_path_prefix='data.input.path.',
                         key_in_dicoms_prefix='data.input.dicoms.',
                         key_out_prefix='data.input.volumes.')),
            # step 4: group DCE sequnces into DCE_mix
            (OpGroupDCESequences(),
                    dict( key_sequence_ids='data.input.sequence_ids',
                     key_path_prefix='data.input.path.',
                     key_series_num_prefix='data.input.series_num.',
                     key_volumes_prefix='data.input.volumes.')),

            # step 5: select single volume from DCE_mix sequence
            (OpSelectVolumes(get_indexes_func=select_series_func),
                    dict(key_in_sequence_ids='data.input.sequence_ids',
                         key_in_path_prefix='data.input.path.',
                         key_in_volumes_prefix='data.input.volumes.',
                         key_out_paths='data.input.selected_paths',
                         key_out_volumes='data.input.selected_volumes')),

            # step 6: set reference volume to be first and register other volumes with respect to it
            (OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline'),
                    dict(key_in='data.input.selected_volumes',
                         key_out='data.input.selected_volumes_resampled')),

            # step 7: create 4D volume from all the sequences
            (OpStackList4DStk(), dict(key_in='data.input.selected_volumes',
                                      key_out_volume4d='data.input.volume4D',
                                      key_out_ref_volume='data.input.ref_volume')),

            # step 8:
            (OpRescale4DStk(), dict(key='data.input.volume4D')),

            (OpCreatePatch(get_annotations_func=get_annotations, lsn_shape=(9, 100, 100), lsn_spacing=(1, 0.5, 0.5)),
                    dict(key_in_volume4d='data.input.volume4D',
                        key_in_ref_volume='data.input.ref_volume',
                        key_out='data.input.patches'))
        ])


        return static_pipeline

    @staticmethod
    def dynamic_pipeline():
        return None

    @staticmethod
    def dataset(cache_dir: str, data_path: Optional[str]=None,
                select_series_func=get_selected_series_index,
                reset_cache: bool = False, num_workers: int = 10,
                sample_ids: Optional[Sequence[Hashable]] = None) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param reset_cache: set to True tp reset the cache
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        """


        if data_path is None:
            data_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

        if sample_ids is None:
            sample_ids = Duke.sample_ids()


        static_pipeline = Duke.static_pipeline(data_path, select_series_func=select_series_func)
        dynamic_pipeline = Duke.dynamic_pipeline()

        cacher = SamplesCacher(f'duke_cache_ver{Duke.DUKE_DATASET_VER}',
                               static_pipeline,
                               [cache_dir], restart_cache=reset_cache, workers=num_workers)

        my_dataset = DatasetDefault(sample_ids=sample_ids,
                                    static_pipeline=static_pipeline,
                                    dynamic_pipeline=dynamic_pipeline,
                                    cacher=None, #cacher, todo: change
                                    )
        my_dataset.create()
        return my_dataset


def _get_sequence_2_series_desc_mapping(metadata_path: str):

    # read metadata file and match between series_desc in metadata file and sequence
    metadata_df = pd.read_csv(metadata_path)
    series_description_list = metadata_df['Series Description'].unique()

    series_desc_2_sequence_mapping = {'ax dyn':'DCE_mix_ph'}

    patterns = ['1st', '2nd', '3rd', '4th']
    for i_phase in range(1,5):
        seq_id = f'DCE_mix_ph{i_phase}'
        series_desc_2_sequence_mapping[seq_id] = []
        phase_patterns = [patterns[i_phase-1], f'{i_phase}ax', f'{i_phase}Ax' ,f'{i_phase}/ax', f'{i_phase}/Ax']

        for series_desc in series_description_list:
            has_match =any(p in series_desc for p in phase_patterns)
            if has_match:

                series_desc2 = series_desc.replace(f'{i_phase}ax', f'{i_phase}/ax').replace(f'{i_phase}Ax', f'{i_phase}/Ax')
                series_desc_2_sequence_mapping[series_desc]=seq_id
                series_desc_2_sequence_mapping[series_desc2] = seq_id

    return series_desc_2_sequence_mapping

