
import os
import sys
from typing import Callable, Optional

from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import aug_op_affine, aug_op_color, aug_op_gaussian
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.sampler.sampler_balanced_batch import FuseSamplerBalancedBatch
from fuse.data.processor.processor_csv import FuseProcessorCSV

from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse_examples.tutorials.multimodality_image_clinical.download import download_and_extract_isic
from torch.utils.data.dataloader import DataLoader

sys.path.append(".")
from .input_processor import FuseSkinInputProcessor
from .ground_truth_processor import FuseSkinGroundTruthProcessor
from .data_source import FuseSkinDataSource

def isic_2019_dataset(data_dir: str = 'data', size: int = None, reset_cache: bool = False, post_cache_processing_func: Optional[Callable] = None):
    #data_dir = "data"
    cache_dir = "cache"
    augmentation_pipeline = [
        [
            ('data.input.image',),
            aug_op_affine,
            {'rotate': Uniform(-180.0, 180.0), 'translate': (RandInt(-50, 50), RandInt(-50, 50)),
            'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
            {'apply': RandBool(0.9)}
        ],
        [
            ('data.input.image',),
            aug_op_color,
            {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
            'contrast': Uniform(0.85, 1.15)},
            {'apply': RandBool(0.7)}
        ],
        [
            ('data.input.image',),
            aug_op_gaussian,
            {'std': 0.03},
            {'apply': RandBool(0.7)}
        ],
    ]
    path = os.path.join(data_dir, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv')
    train_data_source = FuseSkinDataSource(path,
                                           partition_file=os.path.join(data_dir, 'ISIC2019/partition.pickle'),
                                           train=True,
                                           size=size,
                                           override_partition=True)


    input_processors = {
        'image': FuseSkinInputProcessor(input_data=os.path.join(data_dir, 'ISIC2019/ISIC_2019_Training_Input')),
        # 'clinical': FuseSkinClinicalProcessor(input_data=os.path.join(data_dir, 'ISIC2019/ISIC_2019_Training_Metadata.csv'))
        'clinical': FuseProcessorCSV(csv_filename=os.path.join(data_dir, 'ISIC2019/ISIC_2019_Training_Metadata.csv'), sample_desc_column="image")
    }
 
    gt_processors = {
        'gt_global': FuseSkinGroundTruthProcessor(input_data=os.path.join(data_dir, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv'))
    }

    # Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(
        augmentation_pipeline=augmentation_pipeline)

    # Create visualizer (optional)
    visualiser = FuseVisualizerDefault(image_name='data.input.image', label_name='data.gt.gt_global.tensor', metadata_names=["data.input.clinical"], gray_scale=False)

    # Create dataset
    train_dataset = FuseDatasetDefault(cache_dest=cache_dir,
                                       data_source=train_data_source,
                                       input_processors=input_processors,
                                       gt_processors=gt_processors,
                                       post_processing_func=post_cache_processing_func,
                                       augmentor=augmentor,
                                       visualizer=visualiser)

    print(f'- Load and cache data:')
    train_dataset.create(reset_cache=reset_cache)
    
    print(f'- Load and cache data: Done')

    ## Create sampler
    print(f'- Create sampler:')
    sampler = FuseSamplerBalancedBatch(dataset=train_dataset,
                                       balanced_class_name='data.gt.gt_global.tensor',
                                       num_balanced_classes=8,
                                       batch_size=8,
                                       use_dataset_cache=True)

    print(f'- Create sampler: Done')

    ## Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=False, drop_last=False,
                                  batch_sampler=sampler, collate_fn=train_dataset.collate_fn,
                                  num_workers=8)
    print(f'Train Data: Done', {'attrs': 'bold'})

    #### Validation data
    print(f'Validation Data:', {'attrs': 'bold'})

    ## Create data source
    validation_data_source = FuseSkinDataSource(os.path.join(data_dir, 'ISIC2019/ISIC_2019_Training_GroundTruth.csv'),
                                                size=size,
                                                partition_file=os.path.join(data_dir, 'ISIC2019/partition.pickle'),
                                                train=False)


    ## Create dataset
    validation_dataset = FuseDatasetDefault(cache_dest=cache_dir,
                                            data_source=validation_data_source,
                                            input_processors=input_processors,
                                            gt_processors=gt_processors,
                                            post_processing_func=post_cache_processing_func,
                                            visualizer=visualiser)

    print(f'- Load and cache data:')
    validation_dataset.create(pool_type='thread')  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    print(f'- Load and cache data: Done')

    ## Create dataloader
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       shuffle=False,
                                       drop_last=False,
                                       batch_sampler=None,
                                       batch_size=8,
                                       num_workers=8,
                                       collate_fn=validation_dataset.collate_fn)
    print(f'Validation Data: Done', {'attrs': 'bold'})

    return train_dataloader, validation_dataloader


SEX_INDEX = {
    'male': 0,
    'female': 1
}
ANATOM_SITE_INDEX = {
    'anterior torso': 0, 'upper extremity': 1, 'posterior torso': 2,
    'lower extremity': 3, 'lateral torso': 4, 'head/neck': 5,
    'palms/soles': 6, 'oral/genital': 7
}

if __name__ == "__main__":
    download_and_extract_isic(golden_only=True)
    tt, tt2 = isic_2019_dataset(reset_cache=True, size=400)
    tt.dataset.summary(["data.gt.gt_global.tensor"])
    tt.dataset.visualize_augmentation(0)
