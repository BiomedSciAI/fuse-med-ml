import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from fuseimg.datasets import duke
from fuse.data.utils.sample import create_initial_sample
from fuseimg.data.ops import ops_mri
from fuseimg.datasets.duke import Duke,get_duke_raw_annotations_df
from fuse.data import PipelineDefault
import pickle as pkl

import radiomics as radiomics
# from BigMedilytics.BreastRadiomics.Algorithms.MRI_fuse.cv_and_radiomics.various_FCM_segmentation.FCM import FCM

def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
        map = {'DCE_mix': [2], 'MASK': [0]}
    else:
        map = {'DCE_mix': [1], 'MASK': [0]}
    return map[seq_id]

def sort_studies_by_date(sample_dir):
    studies_names = os.listdir(sample_dir)
    delta = [ser_name[0:5] for ser_name in studies_names]
    sorted_inx = np.argsort(delta)
    studies_names_sorted = [studies_names[inx] for inx in sorted_inx]
    if len(sorted_inx) == 2:
        studies_names_sorted.append('NAN')
    if len(sorted_inx) == 1:
        studies_names_sorted.extend(['NAN'] * 2)
    return studies_names_sorted

def create_lesion_prop_list_from_annotation_slices_range(root_path=None,lesion_table_file=None):
    if root_path is None:
        root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
    if lesion_table_file is None:
        lesion_table_file = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/CURIE/DUKE/experiments/new.csv'

    static_pipeline = DukeLesionProp.static_pipeline(root_path=root_path, select_series_func=duke.get_selected_series_index)
    dynamic_pipeline = DukeLesionProp.dynamic_pipeline()
    patient_dict = {}
    cols_list = [
            'centroid_T0', 'bbox_T0', 'volume_T0', 'elongation_T0', 'size_bbox_x_T0', 'size_bbox_y_T0', 'size_bbox_z_T0',
            'ser_name_T0',
            'centroid_T1', 'bbox_T1', 'volume_T1', 'elongation_T1', 'size_bbox_x_T1', 'size_bbox_y_T1', 'size_bbox_z_T1',
            'ser_name_T1',
            'centroid_T2', 'bbox_T2', 'volume_T2', 'elongation_T2', 'size_bbox_x_T2', 'size_bbox_y_T2', 'size_bbox_z_T2',
            'ser_name_T2',
        ]
    sample_ids = DukeLesionProp.sample_ids()#[:2]
    for sample_id in tqdm(sample_ids):
            print(sample_id)
            try:
                sample_dir = root_path +'Duke-Breast-Cancer-MRI/'+ sample_id
                studies_list = sort_studies_by_date(sample_dir)

                cols_list = []
                lesion_prop_list = ()

                for study_inx, study in enumerate(studies_list):
                    try:
                        sample_dict = create_initial_sample(sample_id)
                        sample_dict = static_pipeline(sample_dict)
                        sample_dict = dynamic_pipeline(sample_dict)
                        ser = [study]
                        lesion_prop = sample_dict['data.lesion_prop']
                        cols =  sample_dict['data.lesion_prop_col']
                    except:
                        ser = None
                        cols = ['centroid', 'bbox', 'volume', 'elongation', 'size_bbox_x', 'size_bbox_y', 'size_bbox_z']
                        lesion_prop = [(None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                )]
                        cols = cols + ['ser_name']
                    cols_list = cols_list + [col + '_T' + str(study_inx) for col in cols]
                    lesion_prop_list = lesion_prop_list + lesion_prop[0] + tuple([ser])
                    patient_dict[sample_id] = lesion_prop_list

            except:
                print('No MRI')
                lesion_prop = [(None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                )]
                lesion_prop_list = lesion_prop[0]
                patient_dict[sample_id] = lesion_prop_list[0]

    df_lesions = pd.DataFrame.from_dict(patient_dict, orient='index')
    df_lesions.columns = cols_list
    df_lesions.index.names = ['Patient ID DICOM']
    df_lesions.to_csv(lesion_table_file)

def create_radiomics(root_path=None,radiomics_table_file=None,setting=None, get_selected_series_index=None):

    if setting is None:
        setting = {}

        setting['seq_vec'] = ['DCE']
        setting['seq_inx_vec'] = [0]
        setting['norm_method'] = 'default'
        setting['maskType'] = 'full'

        if setting['norm_method'] == 'default':
            setting['normalize'] = True
            setting['normalizeScale'] = 100
        else:
            setting['normalize'] = False

        setting['binWidth'] = 5
        setting['preCrop'] = True
        setting['applyLog'] = False
        setting['applyWavelet'] = False

    if get_selected_series_index is None:
        get_selected_series_index = duke.get_selected_series_index

    # Instantiate the extractor
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**setting)
    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    if root_path is None:
        root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'
    if radiomics_table_file is None:
        radiomics_table_file = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/my_research/CURIE/DUKE/experiments/new_radiomics.csv'

    static_pipeline = DukeGetRadiomics.static_pipeline(root_path=root_path, select_series_func=get_selected_series_index)
    dynamic_pipeline = DukeGetRadiomics.dynamic_pipeline(extractor,setting)
    patient_dict = {}

    sample_ids = DukeLesionProp.sample_ids()[0:2]
    for sample_id in tqdm(sample_ids):
        print(sample_id)
        try:
            sample_dict = create_initial_sample(sample_id)
            sample_dict = static_pipeline(sample_dict)
            sample_dict = dynamic_pipeline(sample_dict)
            patient_dict[sample_id] = sample_dict['data.radiomics']

        except:
            print('No MRI')
            patient_dict[sample_id] = None

    df_lesions = pd.DataFrame.from_dict(patient_dict, orient='index')
    df_lesions.index.names = ['Patient ID DICOM']
    df_lesions.to_csv(radiomics_table_file)




class DukeLesionProp(Duke):

    @staticmethod
    def sample_ids():
        annotations_df = get_duke_raw_annotations_df()
        return annotations_df['Patient ID'].values
    @staticmethod
    def dynamic_pipeline():

        def get_annotations(sample_id):
            patient_annotations_df = annotations_df[annotations_df['Patient ID'] == sample_id]
            return patient_annotations_df

        annotations_df = get_duke_raw_annotations_df()

        steps = [(ops_mri.OpExtractLesionPropFromBBoxAnotation(get_annotations), dict(key_in_ref_volume='data.input.ref_volume',
                                                                                          key_out_lesion_prop='data.lesion_prop',
                                                                                          key_out_cols='data.lesion_prop_col'))]

        dynamic_pipeline = PipelineDefault("dynamic", steps)

        return dynamic_pipeline

class DukeGetRadiomics(Duke):

    @staticmethod
    def sample_ids():
        annotations_df = get_duke_raw_annotations_df()
        return annotations_df['Patient ID'].values

    @staticmethod
    def static_pipeline(root_path, select_series_func) -> PipelineDefault:
        static_pipline = Duke.static_pipeline(root_path=root_path, select_series_func=select_series_func)
        # remove scaling operator for radiomics calculation
        del static_pipline._op_ids[7]
        del static_pipline._ops_and_kwargs[7]
        return static_pipline

    @staticmethod
    def dynamic_pipeline(extractor,setting):

        steps = [(ops_mri.OpExtractRadiomics(extractor, setting), dict(key_in_vol_4d='data.input.volume4D', key_out_radiomics_results='data.radiomics'))]

        dynamic_pipeline = PipelineDefault("dynamic", steps)

        return dynamic_pipeline


def get_selected_series_index_radiomics(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ['Breast_MRI_120', 'Breast_MRI_596']:
        map = {'DCE_mix': [1,2], 'MASK': [0]}
    else:
        map = {'DCE_mix': [0,1], 'MASK': [0]}
    return map[seq_id]

if __name__ == "__main__":
    # create_lesion_prop_list_from_annotation_slices_range()
    # root_path = '/projects/msieve2/Platform/BigMedilytics/Data/Duke-Breast-Cancer-MRI/manifest-1607053360376/'

    setting = {}

    setting['seq_list'] = ['DCE0','DCE1']
    setting['seq_inx_list'] = [0,1]
    setting['norm_method'] = 'default'
    setting['maskType'] = 'full'

    if setting['norm_method'] == 'default':
        setting['normalize'] = True
        setting['normalizeScale'] = 100
    else:
        setting['normalize'] = False

    setting['binWidth'] = 5
    setting['preCrop'] = True
    setting['applyLog'] = False
    setting['applyWavelet'] = False

    create_radiomics(setting=setting,get_selected_series_index=get_selected_series_index_radiomics)
