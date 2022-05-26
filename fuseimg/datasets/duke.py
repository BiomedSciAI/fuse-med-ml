from functools import partial
import os
from typing import Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import skimage
import skimage.transform


from fuse.utils import NDict
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform


from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuse.data.ops.op_base import OpBase, OpReversibleBase
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_common import OpLambda

from fuse.data.utils.sample import get_sample_id

from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpClip, OpToRange
from fuseimg.data.ops.ops_mri import OpExtractDicomsPerSeq, OpLoadDicomAsStkVol, OpGroupDCESequences, OpSelectVolumes, OpResampleStkVolsBasedRef




class OpDukeSampleIDDecode(OpReversibleBase):
    '''
    decodes sample id into path of MRI images
    '''

    def __init__(self, data_path:str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path

    def __call__(self, sample_dict:NDict, key_out:str, op_id: Optional[str]) -> NDict:

        sid = get_sample_id(sample_dict)

        sample_dict[key_out] = os.path.join(self._data_path, *sid)

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
        TEST_PATIENT_ID, TEST_STUDY_ID = 'Breast_MRI_900', '01-01-1990-BREASTROUTINE DYNAMICS-51487'

        test_sample_id = (TEST_PATIENT_ID, TEST_STUDY_ID)
        return [test_sample_id]

    @staticmethod
    def static_pipeline(root_path: Optional[str]=None, seq_ids: Optional[List]=None,
                        select_series_func=get_selected_series_index) -> PipelineDefault:

        if root_path is None:
            root_path= '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
        if seq_ids is None:
            seq_ids = ['DCE_mix_ph1', 'DCE_mix_ph3']

        data_path = os.path.join(root_path, 'Duke-Breast-Cancer-MRI')
        metadata_path = os.path.join(root_path, 'metadata.csv')
        sequence_2_series_desc_map = _get_sequence_2_series_desc_mapping(metadata_path)

        static_pipeline = PipelineDefault("static", [
            # step 1: map sample_ids to
            (OpDukeSampleIDDecode(data_path=data_path),
                    dict(key_out='data.input.mri_path')),
            # step 2: read files info for the sequences
            (OpExtractDicomsPerSeq(seq_ids=seq_ids, seq_dict=sequence_2_series_desc_map, use_order_indicator=False),
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
            (OpSelectVolumes(subseq_to_use=['DCE_mix'], get_indexes_func=select_series_func),
                    dict(key_in_path_prefix='data.input.path.',
                         key_in_volumes_prefix='data.input.volumes.',
                         key_out_path_prefix='data.input.selected_path.',
                         key_out_volumes_prefix='data.input.selected_volumes.')),

            (OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline'),
                    dict( key_seq_ids='data.input.sequence_ids',
                                                 key_seq_volumes_prefix='data.input.selected_volumes.',
                                                 key_out_prefix='data.input.selected_volumes_resampled.')),

        ])


        return static_pipeline

    @staticmethod
    def dynamic_pipeline():
        return None

    @staticmethod
    def dataset(cache_dir: str, data_path: Optional[str]=None, reset_cache: bool = False, num_workers: int = 10,
                sample_ids: Optional[Sequence[Hashable]] = None) -> DatasetDefault:
        """
        Get cached dataset
        :param data_path: path to store the original data
        :param cache_dir: path to store the cache
        :param reset_cache: set to True tp reset the cache
        :param num_workers: number of processes used for caching
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        """

        if sample_ids is None:
            sample_ids = Duke.sample_ids()

        if data_path is None:
            data_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

        static_pipeline = Duke.static_pipeline(data_path)
        dynamic_pipeline = Duke.dynamic_pipeline()

        cacher = SamplesCacher(f'duke_cache_ver{Duke.DUKE_DATASET_VER}',
                               static_pipeline,
                               [cache_dir], restart_cache=reset_cache, workers=num_workers)

        my_dataset = DatasetDefault(sample_ids=sample_ids,
                                    static_pipeline=static_pipeline,
                                    dynamic_pipeline=dynamic_pipeline,
                                    cacher=cacher,
                                    )
        my_dataset.create()
        return my_dataset


def _get_sequence_2_series_desc_mapping(metadata_path: str):

    # read metadata file and match between series_desc in metadata file and sequence
    metadata_df = pd.read_csv(metadata_path)
    series_description_list = metadata_df['Series Description'].unique()

    sequence_2_series_desc_mapping = {'DCE_mix_ph': ['ax dyn']}
    patterns = ['1st', '2nd', '3rd', '4th']
    for i_phase in range(1,5):
        seq_id = f'DCE_mix_ph{i_phase}'
        sequence_2_series_desc_mapping[seq_id] = []
        phase_patterns = [patterns[i_phase-1], f'{i_phase}ax', f'{i_phase}Ax' ,f'{i_phase}/ax', f'{i_phase}/Ax']

        for series_desc in series_description_list:
            has_match =any(p in series_desc for p in phase_patterns)
            if has_match:
                series_desc2 = series_desc.replace(f'{i_phase}ax', f'{i_phase}/ax').replace(f'{i_phase}Ax', f'{i_phase}/Ax')
                sequence_2_series_desc_mapping[seq_id] += [series_desc, series_desc2]


    return sequence_2_series_desc_mapping

