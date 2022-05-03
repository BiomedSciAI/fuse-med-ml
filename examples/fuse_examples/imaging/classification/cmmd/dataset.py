import sys
from typing import Callable, Optional , Sequence
import logging
import pandas as pd
import pydicom
import os, glob
from pathlib import Path


from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.image_loader import OpLoadImage , OpLoadDicom
from fuseimg.data.ops.color import OpClip, OpToRange
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D , OpAugCropAndResize2D
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.utils.ndict import NDict
import torch
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool

# from fuse_examples.imaging.classification.cmmd.input_processor import MGInputProcessor
# from fuse_examples.imaging.classification.cmmd.ground_truth_processor import MGGroundTruthProcessor
from fuse.data.utils.split import SplitDataset
from tempfile import mkdtemp
from typing import Tuple
from fuse.data.utils.sample import get_sample_id

class OpCmmdDecode(OpBase):
    '''
    decodes sample id into image and segmentation filename
    '''
    """
    Op that extract data from pytorch dataset that returning sequence of values and adds those values to sample_dict
    """

    def __init__(self, df : pd.DataFrame):
        """
        :param dataset: the pytorch dataset to convert. The dataset[i] expected to return sequence of values or a single value
        :param sample_keys: sequence keys - naming each value returned by dataset[i]
        """
        # store input arguments
        super().__init__()
        self._df = df
    def __call__(self, sample_dict: NDict,  op_id: Optional[str]) -> NDict:
        '''
        
        '''
        #sample_keys=('data.image', 'data.label')
        sample_id = get_sample_id(sample_dict)
        sid = self._df.iloc[sample_id]['file']
        
        img_filename_key = 'data.input.img_path'
        sample_dict[img_filename_key] = sid
        return sample_dict
    
def CMMD_2021_dataset(data_dir: str, data_misc_dir: str ,cache_dir: str = 'cache', reset_cache: bool = False,
                      post_cache_processing_func: Optional[Callable] = None) -> Tuple[DatasetDefault, DatasetDefault]:
    """
    Creates Fuse Dataset object for training, validation and test
    :param data_dir:                    dataset root path
    :param data_misc_dir                path to save misc files to be used later
    :param cache_dir:                   Optional, name of the cache folder
    :param reset_cache:                 Optional,specifies if we want to clear the cache first
    :param post_cache_processing_func:  Optional, function run post cache processing
    :return: training, validation and test DatasetDefault objects
    """
    # augmentation_pipeline = [
    #     [
    #         ('data.input.image',),
    #         aug_op_affine,
    #         {'rotate': Uniform(-30.0, 30.0), 'translate': (RandInt(-10, 10), RandInt(-10, 10)),
    #          'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
    #         {'apply': RandBool(0.5)}
    #     ],
    #     [
    #         ('data.input.image',),
    #         aug_op_color,
    #         {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
    #          'contrast': Uniform(0.85, 1.15)},
    #         {'apply': RandBool(0.5)}
    #     ],
    #     [
    #         ('data.input.image',),
    #         aug_op_gaussian,
    #         {'std': 0.03},
    #         {'apply': RandBool(0.5)}
    #     ],
    # ]


    lgr = logging.getLogger('Fuse')
    target = 'classification'
    input_source_gt = merge_clinical_data_with_dicom_tags(data_dir, data_misc_dir, target)
    input_df = pd.read_csv(input_source_gt)
    folds_df = SplitDataset.balanced_division(df = input_df ,
                                                no_mixture_id = 'ID1',
                                                key_columns = [target] ,
                                                nfolds = 5 ,
                                                print_flag=True )
    sample_ids=[id for id in range(len(folds_df))]

    static_pipeline = PipelineDefault("static", [
        # (OpCmmdDecode(folds_df), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
        (OpReadDataframe(folds_df,key_column = None , columns_to_extract = ['file','classification'] , rename_columns=dict(file="data.input.img_path",classification="data.gt.classification")), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" 
        (OpLoadDicom(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
        (OpToRange(), dict(key="data.input.img", from_range=(0, 255), to_range=(0, 1))),
        (OpAugCropAndResize2D(), dict(key="data.input.img", from_range=(0, 255), to_range=(0, 1))),
        ])

    dynamic_pipeline = PipelineDefault("dynamic", [
        (OpToTensor(), dict(dtype=torch.float32)),
        (OpSample(OpAugAffine2D()), dict(
                        rotate=Uniform(-30.0,30.0),        
                        scale=Uniform(0.9, 1.1),
                        flip=(RandBool(0.3), RandBool(0.5)),
                        translate=(RandInt(-10, 10), RandInt(-10, 10))
                    )),
        (OpSample(OpAugColor()), dict(
                    key="data.input.img",
                    gamma=Uniform(0.9, 1.1), 
                    contrast=Uniform(0.85, 1.15),
                    mul =  Uniform(0.95, 1.05),
                    add=Uniform(-0.06, 0.06)
                )),
    ])
                                       
    cache_dir = mkdtemp(prefix="cmmd")
    cacher = SamplesCacher(f'cmmd_cache_ver', 
        static_pipeline,
        cache_dirs=[cache_dir], restart_cache=True)   
    
    my_dataset = DatasetDefault(sample_ids=sample_ids,
        static_pipeline=static_pipeline,
        dynamic_pipeline=dynamic_pipeline,
        cacher=cacher,            
    )


    my_dataset.create()
    
    # partition_file_path = os.path.join(data_misc_dir, 'data_fold_new.csv')
    
    # train_data_source = DataSourceFolds(input_source=input_source_gt,
    #                                         input_df=None,
    #                                         phase='train',
    #                                         no_mixture_id='ID1',
    #                                         balance_keys=[target],
    #                                         reset_partition_file=True,
    #                                         folds=[0,1,2],
    #                                         num_folds=5,
    #                                         partition_file_name=partition_file_path)

    

    # # Create data processors:
    # input_processors = {
    #     'image': MGInputProcessor(input_data=data_dir)
    # }
    # gt_processors = {
    #     'classification': MGGroundTruthProcessor(input_data=input_source_gt)
    # }



    # # Create train dataset
    # train_dataset = DatasetDefault(cache_dest=cache_dir,
    #                                    data_source=train_data_source,
    #                                    input_processors=input_processors,
    #                                    gt_processors=gt_processors,
    #                                    post_processing_func=post_cache_processing_func,
    #                                    augmentor=augmentor)

    # lgr.info(f'- Load and cache data:')
    # train_dataset.create(reset_cache=reset_cache)
    # lgr.info(f'- Load and cache data: Done')

    # # Create validation data source
    # validation_data_source = DataSourceFolds(input_source=input_source_gt,
    #                                         input_df=None,
    #                                         phase='validation',
    #                                         no_mixture_id='ID1',
    #                                         balance_keys=[target],
    #                                         reset_partition_file=False,
    #                                         folds=[3],
    #                                         num_folds=5,
    #                                         partition_file_name=partition_file_path)

    # ## Create dataset
    # validation_dataset = DatasetDefault(cache_dest=cache_dir,
    #                                         data_source=validation_data_source,
    #                                         input_processors=input_processors,
    #                                         gt_processors=gt_processors,
    #                                         post_processing_func=post_cache_processing_func,
    #                                         augmentor=None)
    # validation_dataset.create( pool_type='thread')  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading

    # test_data_source =  DataSourceFolds(input_source=input_source_gt,
    #                                         input_df=None,
    #                                         phase='test',
    #                                         no_mixture_id='ID1',
    #                                         balance_keys=[target],
    #                                         reset_partition_file=False,
    #                                         folds=[4],
    #                                         num_folds=5,
    #                                         partition_file_name=partition_file_path)
    # test_dataset = DatasetDefault(cache_dest=cache_dir,
    #                                         data_source=test_data_source,
    #                                         input_processors=input_processors,
    #                                         gt_processors=gt_processors,
    #                                         post_processing_func=post_cache_processing_func,
    #                                         augmentor=None)
    
    # test_dataset.create( pool_type='thread')  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading

    lgr.info(f'- Load and cache data:')

    lgr.info(f'- Load and cache data: Done')

    return my_dataset, my_dataset, my_dataset


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
        return combined_file_path
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
    merged_clinical_data.to_csv(combined_file_path)
    return combined_file_path

