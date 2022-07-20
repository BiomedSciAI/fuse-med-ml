import gzip
import os
import pickle

import pandas as pd


def save_object(obj, filename):
    open_func = gzip.open if filename.endswith(".gz") else open
    filename_tmp = filename+ ".del"
    if os.path.exists(filename_tmp):
        os.remove(filename_tmp)
    with open_func(filename_tmp, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    os.rename(filename_tmp, filename)
    return filename


def load_object(filename):
    open_func = gzip.open if filename.endswith(".gz") else open

    with open_func(filename, 'rb') as myinput:
        try:
            res = pickle.load(myinput)
        except RuntimeError as e:
            print("Failed to read", filename)
            raise e
    return res


DUKE_PROCESSED_FILE_DIR = '/projects/msieve_dev3/usr/common/duke_processed_files'


def get_duke_annotations_from_tal_df():
    annotations_path = os.path.join(DUKE_PROCESSED_FILE_DIR, 'dataset_DUKE_folds_ver11102021TumorSize_seed1.pickle')
    with open(annotations_path, 'rb') as infile:
        fold_annotations_dict = pickle.load(infile)
    annotations_df = pd.concat(
        [fold_annotations_dict[f'data_fold{fold}'] for fold in range(len(fold_annotations_dict))])
    return annotations_df


def get_col_mapping():
    return {'MRI Findings:Skin/Nipple Invovlement': 'Skin Invovlement',
            'US features:Tumor Size (cm)': 'Tumor Size US',
            'Mammography Characteristics:Tumor Size (cm)': 'Tumor Size MG',
            'MRI Technical Information:FOV Computed (Field of View) in cm': 'Field of View',
            'MRI Technical Information:Contrast Bolus Volume (mL)': 'Contrast Bolus Volume',
            'Demographics:Race and Ethnicity': 'Race',
            'MRI Technical Information:Manufacturer Model Name': 'Manufacturer',
            'MRI Technical Information:Slice Thickness': 'Slice Thickness',
            'MRI Findings:Multicentric/Multifocal': 'Multicentric',
            'Mammography Characteristics:Breast Density': 'Breast Density MG',
            'Tumor Characteristics:PR': 'PR',
            'Tumor Characteristics:HER2': 'HER2',
            'Tumor Characteristics:ER': 'ER',
            'Near Complete Response:Overall Near-complete Response:  Stricter Definition': 'Near pCR Strict',
            'Tumor Characteristics:Staging(Tumor Size)# [T]': 'Staging Tumor Size',
            'Tumor Characteristics:Histologic type': 'Histologic type',
            }
