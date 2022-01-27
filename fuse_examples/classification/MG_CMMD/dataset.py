import sys
from typing import Callable, Optional
import logging
import pandas as pd
import pydicom
import os, glob

from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import aug_op_color, aug_op_gaussian, aug_op_affine
from fuse.data.dataset.dataset_default import FuseDatasetDefault

from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse.utils.utils_logger import fuse_logger_start

from fuse_examples.classification.MG_CMMD.input_processor import FuseMGInputProcessor
from fuse_examples.classification.MG_CMMD.ground_truth_processor import FuseMGGroundTruthProcessor
from fuse.data.data_source.data_source_folds import FuseDataSourceFolds

from typing import Tuple


def CMMD_2021_dataset(data_dir: str, cache_dir: str = 'cache', reset_cache: bool = False,
                      post_cache_processing_func: Optional[Callable] = None) -> Tuple[FuseDatasetDefault, FuseDatasetDefault]:
    """
    Creates Fuse Dataset object for training, validation and test
    :param data_dir:                    dataset root path
    :param cache_dir:                   Optional, name of the cache folder
    :param reset_cache:                 Optional,specifies if we want to clear the cache first
    :param post_cache_processing_func:  Optional, function run post cache processing
    :return: training, validation and test FuseDatasetDefault objects
    """
    augmentation_pipeline = [
        [
            ('data.input.image',),
            aug_op_affine,
            {'rotate': Uniform(-30.0, 30.0), 'translate': (RandInt(-10, 10), RandInt(-10, 10)),
             'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.input.image',),
            aug_op_color,
            {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
             'contrast': Uniform(0.85, 1.15)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.input.image',),
            aug_op_gaussian,
            {'std': 0.03},
            {'apply': RandBool(0.5)}
        ],
    ]

    lgr = logging.getLogger('Fuse')
    training_data = os.path.join(data_dir, 'CMMD/')
    partition_file = "train_val_split.pickle"
    target = 'classification'
    input_source_gt = merge_clinical_data_with_dicom_tags(data_dir, target)

    partition_file_path = os.path.join(data_dir, 'data_fold_new.csv')
    train_data_source = FuseDataSourceFolds(input_source=input_source_gt,
                                            input_df=None,
                                            phase='train',
                                            no_mixture_id='ID1',
                                            balance_keys=[target],
                                            reset_partition_file=True,
                                            folds=[0,1,2],
                                            num_folds=5,
                                            partition_file_name=partition_file_path)
    

    # Create data processors:
    input_processors = {
        'image': FuseMGInputProcessor(input_data=training_data)
    }
    gt_processors = {
        'classification': FuseMGGroundTruthProcessor(input_data=input_source_gt)
    }

    # Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(
        augmentation_pipeline=augmentation_pipeline)

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.image', label_name='data.gt.classification')

    # Create train dataset
    train_dataset = FuseDatasetDefault(cache_dest=cache_dir,
                                       data_source=train_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       post_processing_func=post_cache_processing_func,
                                       augmentor=augmentor,
                                       visualizer=visualiser)

    lgr.info(f'- Load and cache data:')
    train_dataset.create(reset_cache=reset_cache)
    lgr.info(f'- Load and cache data: Done')

    # Create validation data source
    validation_data_source = FuseDataSourceFolds(input_source=input_source_gt,
                                            input_df=None,
                                            phase='validation',
                                            no_mixture_id='ID1',
                                            balance_keys=[target],
                                            reset_partition_file=False,
                                            folds=[3],
                                            num_folds=5,
                                            partition_file_name=partition_file_path)

    ## Create dataset
    validation_dataset = FuseDatasetDefault(cache_dest=cache_dir,
                                            data_source=validation_data_source,
                                            input_processors=input_processors,
                                            gt_processors=gt_processors,
                                            post_processing_func=post_cache_processing_func,
                                            augmentor=None,
                                            visualizer=visualiser)

    test_data_source = FuseDataSourceFolds(input_source_gt,None,'test','ID1',[target],False,[4],5,partition_file_path)

    test_dataset =  FuseDataSourceFolds(input_source=input_source_gt,
                                            input_df=None,
                                            phase='test',
                                            no_mixture_id='ID1',
                                            balance_keys=[target],
                                            reset_partition_file=False,
                                            folds=[4],
                                            num_folds=5,
                                            partition_file_name=partition_file_path)

    lgr.info(f'- Load and cache data:')
    validation_dataset.create(
        pool_type='thread')  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    lgr.info(f'- Load and cache data: Done')

    return train_dataset, validation_dataset, test_dataset


def merge_clinical_data_with_dicom_tags(root_path: str, target: str) -> str:
    """
    Creates a csv file that contains label for each image ( instead of patient as in dataset given file)
    by reading metadata ( breast side and view ) from the dicom files and merging it with the input csv
    If the csv already exists , it will skip the creation proccess
    :param root_path    :       dataset root path
    :return: the new csv file path
    """
    input_source = root_path + '/CMMD_clinicaldata_revision.csv'
    combined_file_path = root_path + '/files_combined.csv'
    if os.path.isfile(combined_file_path):
        return combined_file_path
    clinical_data = pd.read_csv(input_source)
    scans = []
    for patient in os.listdir(root_path + 'CMMD/'):
        path = root_path + 'CMMD/' + patient
        for dicom_file in glob.glob(os.path.join(path, '**/*.dcm'), recursive=True):
            file = os.path.join(path, dicom_file)
            dcm = pydicom.dcmread(file)
            scans.append({'ID1': patient, 'LeftRight': dcm[0x00200062].value, 'file': file,
                          'view': dcm[0x00540220].value.pop(0)[0x00080104].value})
    dicom_tags = pd.DataFrame(scans)
    merged_clinical_data = pd.merge(clinical_data, dicom_tags, how='outer', on=['ID1', 'LeftRight'])
    merged_clinical_data = merged_clinical_data[merged_clinical_data[target].notna()]
    merged_clinical_data.to_csv(combined_file_path)
    return combined_file_path


if __name__ == "__main__":
    # Path to save model
    root = ''
    # Path to store the data
    root_data = '/gpfs/haifa/projects/m/msieve3/CMMD/'
    # Name of the experiment
    experiment = 'InceptionResnetV2_2017_test'
    # Path to cache data
    cache_path = 'examples/'
    # Name of the cached data folder
    experiment_cache = 'CMMD_'
    paths = {'data_dir': root_data,
             'model_dir': os.path.join(root, experiment, 'model_dir'),
             'force_reset_model_dir': True,
             # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
             'cache_dir': None,  # os.path.join(cache_path, experiment_cache + '_cache_dir'),
             'inference_dir': os.path.join(root, experiment, 'infer_dir')}
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths['model_dir'], console_verbose_level=logging.INFO)
    lgr = logging.getLogger('Fuse')

    # Download data
    # TBD

    lgr.info('\nFuse Train', {'attrs': ['bold', 'underline']})

    lgr.info(f'model_dir={paths["model_dir"]}', {'color': 'magenta'})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {'color': 'magenta'})

    #### Train Data
    lgr.info(f'Train Data:', {'attrs': 'bold'})
    tt, tt2 = CMMD_2021_dataset(paths['data_dir'], reset_cache=True)
    # print(tt.summary(["data.gt.classification"]))
    tt.visualize_augmentation(0)
    print("dataset unit test done!")
