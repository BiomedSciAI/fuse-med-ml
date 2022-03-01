import pandas as pd
from functools import partial
from multiprocessing import Manager
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import unsqueeze_2d_to_3d, aug_op_color, aug_op_affine, squeeze_3d_to_2d, \
    rotation_in_3d
from fuse.data.dataset.dataset_generator import FuseDatasetGenerator

from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandBool as RandBool
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerRandInt as RandInt
from fuse.utils.utils_param_sampler import FuseUtilsParamSamplerUniform as Uniform
from fuse.data.visualizer.visualizer_default_3d import Fuse3DVisualizerDefault

from fuse.data.processor.processor_dicom_mri import FuseDicomMRIProcessor

from fuse_examples.classification.prostate_x.patient_data_source import FuseProstateXDataSourcePatient


from fuse_examples.classification.duke_breast_cancer.post_processor import post_processing
from fuse_examples.classification.duke_breast_cancer.processor import FusePatchProcessor


def process_mri_series(metadata_path: str):

    seq_to_use = ['DCE_mix_ph1',
                  'DCE_mix_ph2',
                  'DCE_mix_ph3',
                  'DCE_mix_ph4',
                  'DCE_mix',
                  'DCE_mix_ph',
                  'MASK']
    subseq_to_use = ['DCE_mix_ph2', 'MASK']

    l_seq = pd.read_csv(metadata_path)
    seq_to_use_full = list(l_seq['Series Description'].value_counts().keys())

    SER_INX_TO_USE = {}
    SER_INX_TO_USE['all'] = {'DCE_mix': [1], 'MASK': [0]}
    SER_INX_TO_USE['Breast_MRI_120'] = {'DCE_mix': [2], 'MASK': [0]}
    SER_INX_TO_USE['Breast_MRI_596'] = {'DCE_mix': [2], 'MASK': [0]}
    exp_patients = ['Breast_MRI_120','Breast_MRI_596']
    opt_seq = [
        '1st','1ax','1Ax','1/ax',
        '2nd','2ax','2Ax','2/ax',
        '3rd','3ax','3Ax', '3/ax',
        '4th','4ax','4Ax','4/ax',
    ]
    my_keys = ['DCE_mix_ph1'] * 4 + ['DCE_mix_ph2'] * 4 + ['DCE_mix_ph3'] * 4 + ['DCE_mix_ph4'] * 4
    seq_to_use_full_slash = [s.replace('ax','/ax') for s in seq_to_use_full]
    seq_to_use_full_slash = [s.replace('Ax', '/Ax') for s in seq_to_use_full_slash]

    seq_to_use_dict = {}
    for opt_seq_tmp, my_key in zip(opt_seq, my_keys):
        tt = [s for s in seq_to_use_full if opt_seq_tmp in s]+[s for s in seq_to_use_full_slash if opt_seq_tmp in s]

        for tmp in tt:
            seq_to_use_dict[tmp] = my_key
    seq_to_use_dict['ax dyn'] = 'DCE_mix_ph'
    return seq_to_use_dict,SER_INX_TO_USE,exp_patients,seq_to_use,subseq_to_use



def duke_breast_cancer_dataset(paths,train_common_params,lgr):
    # ==============================================================================
    # Data
    # ==============================================================================
    #### Train Data

    lgr.info(f'Train Data:', {'attrs': 'bold'})

    ## Create data source:
    DATABASE_REVISION = train_common_params['partition_version']
    lgr.info(f'database_revision={DATABASE_REVISION}', {'color': 'magenta'})

    # create data source
    train_data_source = FuseProstateXDataSourcePatient(paths['data_dir'], 'train',
                                                       db_ver=train_common_params['partition_version'],
                                                       db_name=train_common_params['db_name'],
                                                       fold_no=train_common_params['fold_no'])

    ## Create data processors:
    image_processing_args = {
        'patch_xy': 100,
        'patch_z': 9,
    }

    ## Create data processor
    #########################################################################################
    seq_dict, SER_INX_TO_USE, exp_patients,seq_to_use,subseq_to_use = \
                                                    process_mri_series(paths['metadata_path'])
    mri_vol_processor = FuseDicomMRIProcessor(seq_dict=seq_dict,
                                              seq_to_use=seq_to_use,
                                              subseq_to_use=subseq_to_use,
                                              ser_inx_to_use=SER_INX_TO_USE,
                                              exp_patients=exp_patients,
                                              reference_inx=0,
                                              use_order_indicator=False)

    generate_processor = FusePatchProcessor(
        vol_processor=mri_vol_processor,
        path_to_db=paths['data_dir'],
        data_path=paths['data_path'],
        ktrans_data_path=paths['ktrans_path'],
        db_name=train_common_params['db_name'],
        db_version=train_common_params['partition_version'],
        fold_no=train_common_params['fold_no'],
        lsn_shape=(image_processing_args['patch_z'],
                   image_processing_args['patch_xy'],
                   image_processing_args['patch_xy']),
    )

    train_post_processor = partial(post_processing, label=train_common_params['classification_task'])

    # data augmentation (optional)
    num_channels = train_common_params['backbone_model_dict']['input_channels_num'] + 1
    slice_num = image_processing_args['patch_z']

    _no_aug = [list(range(0, slice_num))]
    aug_pipeline = [
        [
            ('data.input',),
            rotation_in_3d,
            {'z_rot': Uniform(-5.0, 5.0), 'y_rot': Uniform(-5.0, 5.0), 'x_rot': Uniform(-5.0, 5.0)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.input',),
            squeeze_3d_to_2d,
            {'axis_squeeze': 'z'},
            {}
        ],
        [
            ('data.input',),
            aug_op_affine,
            {'rotate': Uniform(0, 360.0),
             'translate': (RandInt(-4, 4), RandInt(-4, 4)),
             'flip': (RandBool(0.5), RandBool(0.5)),
             'scale': Uniform(0.9, 1.1),
             },
            {'apply': RandBool(0.5)}
        ],

        [
            ('data.input',),
            unsqueeze_2d_to_3d,
            {'channels': num_channels, 'axis_squeeze': 'z'},
            {}
        ],
    ]
    augmentor = FuseAugmentorDefault(augmentation_pipeline=aug_pipeline)

    visualizer = Fuse3DVisualizerDefault(image_name='data.input', label_name='data.isLargeTumorSize')
    # Create dataset
    train_dataset = FuseDatasetGenerator(cache_dest=paths['cache_dir'],
                                         data_source=train_data_source,
                                         processor=generate_processor,
                                         post_processing_func=train_post_processor,
                                         augmentor=augmentor,
                                         statistic_keys=['data.ground_truth'],
                                         visualizer=visualizer,
                                         )

    lgr.info(f'- Load and cache data:')
    train_dataset.create()

    train_dataset.filter('data.filter', [True])
    lgr.info(f'- Load and cache data: Done')

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create data source
    validation_data_source = FuseProstateXDataSourcePatient(paths['data_dir'], 'validation',
                                                            db_ver=DATABASE_REVISION,
                                                            db_name=train_common_params['db_name'],
                                                            fold_no=train_common_params['fold_no'])

    # post processor
    validation_post_processor = partial(post_processing, label=train_common_params['classification_task'])

    ## Create dataset
    validation_dataset = FuseDatasetGenerator(cache_dest=paths['cache_dir'],
                                              data_source=validation_data_source,
                                              processor=generate_processor,
                                              post_processing_func=validation_post_processor,
                                              augmentor=None,
                                              statistic_keys=['data.ground_truth'],
                                              visualizer=None
                                              )

    lgr.info(f'- Load and cache data:')

    validation_dataset.create(num_workers=0)
    lgr.info(f'Data - task caching and filtering:', {'attrs': 'bold'})

    validation_dataset.filter('data.filter', [True])

    return train_dataset,validation_dataset