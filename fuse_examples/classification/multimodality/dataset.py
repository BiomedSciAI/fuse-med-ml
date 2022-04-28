import sys
from typing import Callable, Optional
import logging
import pandas as pd
import pydicom
import os, glob
from pathlib import Path
from typing import Tuple

from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import aug_op_color, aug_op_gaussian, aug_op_affine
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.dataset.dataset_generator import FuseDatasetGenerator
from fuse.data.data_source.data_source_default import FuseDataSourceDefault

from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool


from fuse_examples.classification.multimodality.input_processor import ImagingTabularProcessor





def IMAGING_dataset():
    """
    Creates Fuse Dataset object for training, validation and test
    :param data_dir:                    dataset root path
    :param data_misc_dir                path to save misc files to be used later
    :param cache_dir:                   Optional, name of the cache folder
    :param reset_cache:                 Optional,specifies if we want to clear the cache first
    :param post_cache_processing_func:  Optional, function run post cache processing
    :return: training, validation and test FuseDatasetDefault objects
    """
    augmentation_pipeline = [
        [
            ('data.image',),
            aug_op_affine,
            {'rotate': Uniform(-30.0, 30.0), 'translate': (RandInt(-10, 10), RandInt(-10, 10)),
             'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.image',),
            aug_op_color,
            {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
             'contrast': Uniform(0.85, 1.15)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.image',),
            aug_op_gaussian,
            {'std': 0.03},
            {'apply': RandBool(0.5)}
        ],
    ]



    # Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(
        augmentation_pipeline=augmentation_pipeline)




    return augmentor


def TABULAR_dataset(tabular_processor,df,tabular_features,sample_key):
    tabular_features.remove(sample_key)
    tabular_processor = tabular_processor(data=df,
                                          sample_desc_column=sample_key,
                                          columns_to_extract=tabular_features + [sample_key],
                                          columns_to_tensor=tabular_features)
    return tabular_processor


def IMAGING_TABULAR_dataset(df, imaging_processor, tabular_processor,label_key:str,img_key:str,tabular_features_lst: list,sample_key: str,
                             cache_dir: str = 'cache', reset_cache: bool = False,
                             post_cache_processing_func: Optional[Callable] = None) -> Tuple[FuseDatasetDefault, FuseDatasetDefault]:


    lgr = logging.getLogger('Fuse')

    if isinstance(df,list):
        df_train = df[0]
        if len(df)>1:
            df_val = df[1]
        if len(df)>2:
            df_test = df[2]

    #----------------------------------------------
    # -----Datasource
    train_data_source = FuseDataSourceDefault(input_source=df_train)
    validation_data_source = FuseDataSourceDefault(input_source=df_val)
    test_data_source = FuseDataSourceDefault(input_source=df_test)

    # ----------------------------------------------
    # -----Data-processors
    img_clinical_processor_train = ImagingTabularProcessor(data=df_train,
                                                           label=label_key,
                                                           img_key = img_key,
                                                           image_processor=imaging_processor(''),
                                                           tabular_processor= \
                                                           TABULAR_dataset(tabular_processor,df_train,tabular_features_lst.copy(),sample_key))

    img_clinical_processor_val = ImagingTabularProcessor(data=df_val,
                                                         label=label_key,
                                                         img_key=img_key,
                                                         image_processor=imaging_processor(''),
                                                         tabular_processor=\
                                                         TABULAR_dataset(tabular_processor,df_val,tabular_features_lst.copy(),sample_key))

    img_clinical_processor_test = ImagingTabularProcessor(data=df_test,
                                                          label=label_key,
                                                          img_key=img_key,
                                                          image_processor=imaging_processor(''),
                                                          tabular_processor= \
                                                          TABULAR_dataset(tabular_processor,df_test,tabular_features_lst.copy(),sample_key))



    visualiser = FuseVisualizerDefault(image_name='data.image', label_name='data.gt')


    # ----------------------------------------------
    # ------ Dataset
    train_dataset = FuseDatasetGenerator(cache_dest=cache_dir,
                                       data_source=train_data_source,
                                       processor=img_clinical_processor_train,
                                       augmentor=IMAGING_dataset(),
                                       visualizer=visualiser,
                                       post_processing_func=post_cache_processing_func,)


    validation_dataset = FuseDatasetGenerator(cache_dest=cache_dir,
                                       data_source=validation_data_source,
                                       processor=img_clinical_processor_val,
                                       augmentor=None,
                                       visualizer=visualiser,
                                       post_processing_func=post_cache_processing_func,)

    test_dataset = FuseDatasetGenerator(cache_dest=cache_dir,
                                       data_source=test_data_source,
                                       processor=img_clinical_processor_test,
                                       augmentor=None,
                                       visualizer=visualiser,
                                       post_processing_func=post_cache_processing_func,)


    # ----------------------------------------------
    # ------ Cache

    # create cache
    train_dataset.create(reset_cache=reset_cache)  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    validation_dataset.create()  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    test_dataset.create()  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading

    lgr.info(f'- Load and cache data:')

    lgr.info(f'- Load and cache data: Done')

    return train_dataset, validation_dataset, test_dataset


