import logging
import pandas as pd
import pydicom
import os, glob
from pathlib import Path

from fuse.data.datasets.dataset_default import DatasetDefault
from fuseimg.datasets.cmmd import CMMD
import numpy as np
from fuse.data.utils.split import SplitDataset
from typing import Tuple, List

def create_folds(input_source: str,
                input_df : pd.DataFrame,
                phase: str,
                no_mixture_id: str,
                balance_keys: np.ndarray,
                reset_partition_file: bool,
                folds: Tuple[int],
                num_folds : int =5,
                partition_file_name: str = None
                ):

    """
    Create DataSource which is divided to num_folds folds, supports either a path to a csv or data frame as input source.
    The function creates a partition file which saves the fold partition
    :param input_source:       path to dataframe containing the samples ( optional )
    :param input_df:           dataframe containing the samples ( optional )
    :param no_mixture_id:      The key column for which no mixture between folds should be forced
    :param balance_keys:       keys for which balancing is forced
    :param reset_partition_file: boolean flag which indicate if we want to reset the partition file
    :param folds               indicates which folds we want to retrieve from the fold partition
    :param num_folds:          number of folds to divide the data
    :param partition_file_name:name of a csv file for the fold partition
                                If train = True, train/val indices are dumped into the file,
                                If train = False, train/val indices are loaded
    :param phase:              specifies if we are in train/validation/test/all phase
    """
    if reset_partition_file is True and phase not in ['train','all']:
        raise Exception("Sorry, it is possible to reset partition file only in train / all phase")
    if reset_partition_file is True or not os.path.isfile(partition_file_name):
        # Load csv file
            # ----------------------

            if input_source is not None :
                input_df = pd.read_csv(input_source)
                create_folds.folds_df = SplitDataset.balanced_division(df = input_df ,
                                                                    no_mixture_id = no_mixture_id,
                                                                    key_columns = balance_keys ,
                                                                    nfolds = num_folds ,
                                                                    print_flag=True )
                # Extract entities
                # ----------------
            else:
                create_folds.folds_df = pd.read_csv(partition_file_name)

    return create_folds.folds_df[create_folds.folds_df['fold'].isin(folds)]


    
def CMMD_2021_dataset(data_dir: str, data_misc_dir: str ,cache_dir: str = 'cache', reset_cache: bool = False) -> Tuple[DatasetDefault, DatasetDefault, DatasetDefault]:
    """
    Creates Fuse Dataset object for training, validation and test
    :param data_dir:                    dataset root path
    :param data_misc_dir                path to save misc files to be used later
    :param cache_dir:                   Optional, name of the cache folder
    :param reset_cache:                 Optional,specifies if we want to clear the cache first
    :return: training, validation and test DatasetDefault objects
    """

    lgr = logging.getLogger('Fuse')
    target = 'classification'
    input_source_gt = merge_clinical_data_with_dicom_tags(data_dir, data_misc_dir, target)
    partition_file_path = os.path.join(data_misc_dir, 'data_fold_new.csv')
    
    lgr.info(f'- Load and cache data:')
    
    train_data_source = create_folds(input_source=input_source_gt,
                                            input_df=None,
                                            phase='train',
                                            no_mixture_id='ID1',
                                            balance_keys=[target],
                                            reset_partition_file=False,
                                            folds=[0,1,2],
                                            num_folds=5,
                                            partition_file_name=partition_file_path)
    train_dataset = CMMD.create_dataset_partition('train',data_dir,train_data_source, cache_dir, reset_cache )
    
    validation_data_source = create_folds(input_source=input_source_gt,
                                            input_df=None,
                                            phase='validation',
                                            no_mixture_id='ID1',
                                            balance_keys=[target],
                                            reset_partition_file=False,
                                            folds=[3],
                                            num_folds=5,
                                            partition_file_name=partition_file_path)
    validation_dataset = CMMD.create_dataset_partition('validation',data_dir,validation_data_source, cache_dir, False)
    
    test_data_source = create_folds(input_source=input_source_gt,
                                            input_df=None,
                                            phase='test',
                                            no_mixture_id='ID1',
                                            balance_keys=[target],
                                            reset_partition_file=False,
                                            folds=[4],
                                            num_folds=5,
                                            partition_file_name=partition_file_path)
    
    test_dataset = CMMD.create_dataset_partition('test',data_dir,test_data_source, cache_dir, False )


    lgr.info(f'- Load and cache data: Done')

    return train_dataset, validation_dataset, test_dataset


def merge_clinical_data_with_dicom_tags(data_dir: str, data_misc_dir:str, target: str) -> str:
    """
    Creates a csv file that contains label for each image ( instead of patient as in dataset given file)
    by reading metadata ( breast side and view ) from the dicom files and merging it with the input csv
    If the csv already exists , it will skip the creation proccess
    :param data_dir                     dataset root path
    :param data_misc_dir                path to save misc files to be used later
    :return: the new csv file path
    """
    input_source = os.path.join(data_dir, 'CMMD_clinicaldata_revision.csv')
    combined_file_path = os.path.join(data_misc_dir, 'files_combined.csv')
    if os.path.isfile(combined_file_path):
        print("Found partition file:",combined_file_path )
        return combined_file_path
    print("Did not find exising partition file!")
    Path(data_misc_dir).mkdir(parents=True, exist_ok=True)
    clinical_data = pd.read_csv(input_source)
    scans = []
    for patient in os.listdir(os.path.join(data_dir,'CMMD')):
        path = os.path.join(data_dir, 'CMMD', patient)
        for dicom_file in glob.glob(os.path.join(path, '**/*.dcm'), recursive=True):
            file = dicom_file[len(data_dir) + 1:] if dicom_file.startswith(data_dir) else ''
            dcm = pydicom.dcmread(os.path.join(data_dir, file))
            scans.append({'ID1': patient, 'LeftRight': dcm[0x00200062].value, 'file': file,
                          'view': dcm[0x00540220].value.pop(0)[0x00080104].value})
    dicom_tags = pd.DataFrame(scans)
    merged_clinical_data = pd.merge(clinical_data, dicom_tags, how='outer', on=['ID1', 'LeftRight'])
    merged_clinical_data = merged_clinical_data[merged_clinical_data[target].notna()]
    merged_clinical_data.classification = np.where(merged_clinical_data.classification == 'Benign', 0, 1)
    merged_clinical_data.to_csv(combined_file_path)
    return combined_file_path