import os
from typing import Hashable, Optional, Sequence
import glob
import os
import pickle
import numpy as np
from typing import Hashable, Optional, Sequence

import pandas as pd

from fuse.data import DatasetDefault
from fuse.data import PipelineDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.op_base import OpReversibleBase, OpBase
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuseimg.data.ops import ops_mri

from enum import Enum, auto

import torch

DUKE_PROCESSED_FILE_DIR = '/projects/msieve_dev3/usr/common/duke_processed_files'

def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
        map = {'DCE_mix': [2], 'MASK': [0]}
    else:
        map = {'DCE_mix': [1], 'MASK': [0]}
    return map[seq_id]

class DukeLabelType(Enum):
    ispCR = 'ispCR'
    STAGING_TUMOR_SIZE = 'Staging Tumor Size'
    HISTOLOGY_TYPE = 'Histology Type'
    IS_HIGH_TUMOR_GRADE_TOTAL = 'is High Tumor Grade Total'

    def get_features_to_use(self):
        if self == DukeLabelType.ispCR:
            return ['Skin Invovlement', 'Tumor Size US', 'Tumor Size MG', 'Field of View',
                           'Contrast Bolus Volume',
                           'Race', 'Manufacturer', 'Slice Thickness']
        if self == DukeLabelType.STAGING_TUMOR_SIZE:
            return ['Skin Invovlement', 'Tumor Size US', 'Tumor Size MG', 'Field of View',
                           'Contrast Bolus Volume',
                           'Race', 'Manufacturer', 'Slice Thickness']
        if self == DukeLabelType.HISTOLOGY_TYPE:
            return ['Skin Invovlement', 'Tumor Size US', 'Tumor Size MG', 'Field of View',
                           'Contrast Bolus Volume',
                           'Race', 'Multicentric', 'Manufacturer', 'Slice Thickness']
        if self == DukeLabelType.IS_HIGH_TUMOR_GRADE_TOTAL:
            return ['Breast Density MG', 'PR', 'HER2', 'ER']
        raise NotImplementedError(self)

    def get_column_name(self):
        if self == DukeLabelType.ispCR:
            return 'Near pCR Strict'
        if self == DukeLabelType.STAGING_TUMOR_SIZE:
            return 'Staging Tumor Size'
        if self == DukeLabelType.HISTOLOGY_TYPE:
            return 'Histologic type'
        if self == DukeLabelType.IS_HIGH_TUMOR_GRADE_TOTAL:
             return 'Tumor Grade Total'
        raise NotImplementedError(self)

    def get_process_func(self):
        if self == DukeLabelType.ispCR:
            def update_func(ispCR):
                if ispCR>2:
                    return np.NaN
                if ispCR == 0 or ispCR == 2:
                    return 0
                else:
                    return 1
            return update_func

        if self == DukeLabelType.STAGING_TUMOR_SIZE:
            return lambda val: 1 if val > 1 else 0

        if self == DukeLabelType.HISTOLOGY_TYPE:
            def update_func(histology_type):
                if histology_type == 1:
                    return 0
                elif histology_type == 10:
                    return 1
                else:
                    return np.NaN
            return update_func

        if self == DukeLabelType.IS_HIGH_TUMOR_GRADE_TOTAL:
            return lambda grade: 1 if grade >= 7 else 0
        raise NotImplementedError(self)


    def get_value(self, clinical_features):
        col_name = self.get_column_name()
        value = clinical_features[col_name] + 0 # why should be add 0??
        process_func = self.get_process_func()
        if process_func is None:
            return value
        if isinstance(value, pd.Series):
            value = value.apply(process_func)
        else:
            value = process_func(value)
        return value

    def get_num_classes(self):
        return 2 # currrently all are binary classification tasks

class Duke:
    DUKE_DATASET_VER = 0

    @staticmethod
    def sample_ids():
        annotations_df = get_duke_annotations_df()
        return annotations_df['Patient ID'].values
        # return [f'Breast_MRI_{i:03d}' for i in range(1,923)]

    @staticmethod
    def static_pipeline(root_path, select_series_func) -> PipelineDefault:

        data_path = os.path.join(root_path, 'Duke-Breast-Cancer-MRI')
        metadata_path = os.path.join(root_path, 'metadata.csv')
        annotations_df = get_duke_annotations_df()

        def get_annotations(sample_id):
            patient_annotations = annotations_df[annotations_df['Patient ID'] == sample_id]
            assert patient_annotations.shape[0] == 1
            return patient_annotations.iloc[0]

        series_desc_2_sequence_map = get_series_desc_2_sequence_mapping(metadata_path)
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
            (ops_mri.OpExtractDicomsPerSeq(seq_ids=seq_ids, series_desc_2_sequence_map=series_desc_2_sequence_map,
                                           use_order_indicator=False),
             dict(key_in='data.input.mri_path',
                  key_out_sequences='data.input.sequence_ids',
                  key_out_path_prefix='data.input.path.',
                  key_out_dicoms_prefix='data.input.dicoms.',
                  key_out_series_num_prefix='data.input.series_num.')
             ),
            # step 3: Load STK volumes of MRI sequences
            (ops_mri.OpLoadDicomAsStkVol(reverse_order=False, is_file=False),
             dict(key_in_seq_ids='data.input.sequence_ids',
                  key_in_path_prefix='data.input.path.',
                  key_in_dicoms_prefix='data.input.dicoms.',
                  key_out_prefix='data.input.volumes.')),
            # step 4: group DCE sequnces into DCE_mix
            (ops_mri.OpGroupDCESequences(),
             dict(key_sequence_ids='data.input.sequence_ids',
                  key_path_prefix='data.input.path.',
                  key_series_num_prefix='data.input.series_num.',
                  key_volumes_prefix='data.input.volumes.')),

            # step 5: select single volume from DCE_mix sequence
            (ops_mri.OpSelectVolumes(get_indexes_func=select_series_func, delete_input_volumes=True),
             dict(key_in_sequence_ids='data.input.sequence_ids',
                  key_in_path_prefix='data.input.path.',
                  key_in_volumes_prefix='data.input.volumes.',
                  key_out_paths='data.input.selected_paths',
                  key_out_volumes='data.input.selected_volumes')),

            # step 6: set reference volume to be first and register other volumes with respect to it
            (ops_mri.OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline'),
             dict(key='data.input.selected_volumes')),

            # step 7: create a single 4D volume from all the sequences (4th channel is the sequence)
            (ops_mri.OpStackList4DStk(delete_input_volumes=True), dict(key_in='data.input.selected_volumes',
                                                                       key_out_volume4d='data.input.volume4D',
                                                                       key_out_ref_volume='data.input.ref_volume')),

            # step 8:
            (ops_mri.OpRescale4DStk(), dict(key='data.input.volume4D')),

            # step 9: read tabular data for each patch
            (ops_mri.OpAddPatchesData(get_annotations_func=get_annotations),
             dict(key_out='data.input.patch_tab_data')),

            # step 10: create patch volumes: (i) fixed size around center of annotatins (orig), and (ii) entire annotations
            (ops_mri.OpCreatePatchVolumes(lsn_shape=(9, 100, 100), lsn_spacing=(1, 0.5, 0.5), delete_input_volumes=True),
             dict(key_in_volume4D='data.input.volume4D',
                  key_in_ref_volume='data.input.ref_volume',
                  key_in_patch_row='data.input.patch_tab_data',
                  key_out_cropped_vol='data.input.patch_volume_orig',
                  key_out_cropped_vol_by_mask='data.input.patch_volume'))
        ])

        return static_pipeline

    @staticmethod
    def dynamic_pipeline(label_type: Optional[DukeLabelType]=None):

        steps = [ (ops_mri.OpStk2Torch(),
                   dict(keys=['data.input.patch_volume_orig','data.input.patch_volume']))]
        if label_type is not None: #todo: wasn't checked!!!
            steps.append((OpAddDukeLabelAndClinicalFeatures(label_type=label_type),
                            dict(key_in='data.input.patch_tab_data',  key_out_gt='data.ground_truth',
                                 key_out_clinical_features='data.clinical_features', key_out_filter='data.filter')))
        dynamic_pipeline = PipelineDefault("dynamic", steps)

        return dynamic_pipeline


    @staticmethod
    def dataset(label_type: Optional[DukeLabelType]=None, cache_dir: Optional[str]=None, data_dir: Optional[str] = None,
                select_series_func=get_selected_series_index,
                reset_cache: bool = False, num_workers: int = 10,
                sample_ids: Optional[Sequence[Hashable]] = None) -> DatasetDefault:

        """
        :param label_type: type of label to use
        :param cache_dir: path to store the cache of the static pipeline
        :param data_dir: path to the original data
        :param select_series_func: which series to select for DCE_mix sequences
        :param reset_cache:
        :param num_workers:  number of processes used for caching
        :param sample_ids: list of selected patient_ids for the dataset
        :return:
        """

        if data_dir is None:
            data_dir = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

        if sample_ids is None:
            sample_ids = Duke.sample_ids()

        static_pipeline = Duke.static_pipeline(data_dir, select_series_func=select_series_func)
        dynamic_pipeline = Duke.dynamic_pipeline(label_type)

        if cache_dir is None:
            cacher = None
        else:
            cacher = SamplesCacher(f'duke_cache_ver{Duke.DUKE_DATASET_VER}',
                                   static_pipeline,
                                   [cache_dir], restart_cache=reset_cache, workers=num_workers,
                                   ignore_nan_inequality=True)

        my_dataset = DatasetDefault(sample_ids=sample_ids,
                                    static_pipeline=static_pipeline,
                                    dynamic_pipeline=dynamic_pipeline,
                                    cacher=cacher
                                    )
        my_dataset.create()
        return my_dataset


class OpDukeSampleIDDecode(OpReversibleBase):
    '''
    decodes sample id into path of MRI images
    '''

    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path

    def __call__(self, sample_dict: NDict, key_out: str, op_id: Optional[str]) -> NDict:
        sid = get_sample_id(sample_dict)

        sample_dict[key_out] = get_sample_path(self._data_path, sid)

        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict



class OpAddDukeLabelAndClinicalFeatures(OpBase):
    '''
    decodes sample id into path of MRI images
    '''

    def __init__(self, label_type: DukeLabelType, is_concat_features_to_input: Optional[bool]=False, **kwargs):
        super().__init__(**kwargs)
        self._label_type = label_type
        self._is_concat_features_to_input = is_concat_features_to_input

    def __call__(self, sample_dict: NDict, key_in: str,
                 key_out_gt: str, key_out_clinical_features: str, key_out_filter:str) -> NDict:
        clinical_features = sample_dict[key_in]
        label_val = self._label_type.get_value(clinical_features)
        if np.isnan(label_val):
            return None # should filter example (instead of sample_dict['data.filter'] = True )

        sample_dict[key_out_filter] = False
        label_tensor = torch.tensor(label_val, dtype=torch.int64)
        sample_dict[key_out_gt] = label_tensor #'data.ground_truth'

        # add clinical
        features_to_use = self._label_type.get_features_to_use()


        clinical_features_to_use = torch.tensor([float(clinical_features[feature]) for feature in features_to_use],
                                                dtype=torch.float32)
        sample_dict[key_out_clinical_features] = clinical_features_to_use # 'data.clinical_features'

        if self._is_concat_features_to_input:
            # select input channel
            input_tensor = sample_dict['data.input']
            input_shape = input_tensor.shape
            for feature in clinical_features_to_use:
                input_tensor = torch.cat(
                    (input_tensor, feature.repeat(input_shape[1], input_shape[2], input_shape[3]).unsqueeze(0)), dim=0)
            sample_dict['data.input'] = input_tensor

        return sample_dict



def get_duke_annotations_df():  # todo: change!!!
    annotations_path = os.path.join(DUKE_PROCESSED_FILE_DIR, 'dataset_DUKE_folds_ver11102021TumorSize_seed1.pickle')
    with open(annotations_path, 'rb') as infile:
        fold_annotations_dict = pickle.load(infile)
    annotations_df = pd.concat(
        [fold_annotations_dict[f'data_fold{fold}'] for fold in range(len(fold_annotations_dict))])
    return annotations_df

def get_samples_for_debug(n_pos, n_neg, label_type):
    annotations_df = get_duke_annotations_df()
    label_values = label_type.get_value(annotations_df)
    patient_ids = annotations_df['Patient ID']
    sample_ids = []
    for label_val, n_vals in  zip([True, False], [n_pos, n_neg]):
        sample_ids += patient_ids[label_values == label_val].values.tolist()[:n_vals]
    return sample_ids
def get_series_desc_2_sequence_mapping(metadata_path: str):
    # read metadata file and match between series_desc in metadata file and sequence
    metadata_df = pd.read_csv(metadata_path)
    series_description_list = metadata_df['Series Description'].unique()

    series_desc_2_sequence_mapping = {'ax dyn': 'DCE_mix_ph'}

    patterns = ['1st', '2nd', '3rd', '4th']
    for i_phase in range(1, 5):
        seq_id = f'DCE_mix_ph{i_phase}'
        phase_patterns = [patterns[i_phase - 1], f'{i_phase}ax', f'{i_phase}Ax', f'{i_phase}/ax', f'{i_phase}/Ax']

        for series_desc in series_description_list:
            has_match = any(p in series_desc for p in phase_patterns)
            if has_match:
                series_desc2 = series_desc.replace(f'{i_phase}ax', f'{i_phase}/ax').replace(f'{i_phase}Ax',
                                                                                            f'{i_phase}/Ax')
                series_desc_2_sequence_mapping[series_desc] = seq_id
                series_desc_2_sequence_mapping[series_desc2] = seq_id

    return series_desc_2_sequence_mapping


def get_sample_path(data_path, sample_id):
    sample_path_pattern = os.path.join(data_path, sample_id, '*')
    sample_path = glob.glob(sample_path_pattern)
    assert len(sample_path) == 1
    return sample_path[0]






