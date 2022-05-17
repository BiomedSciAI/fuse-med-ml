from functools import partial
from multiprocessing import Manager

from fuse.data.augmentor.augmentor_default import AugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import unsqueeze_2d_to_3d, aug_op_color, aug_op_affine, squeeze_3d_to_2d, \
    rotation_in_3d
from fuse.data.dataset.dataset_generator import DatasetGenerator

import fuse.utils.gpu as GPU
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool, Choice

from fuse_examples.imaging.classification.prostate_x.patient_data_source import ProstateXDataSourcePatient
from fuse_examples.imaging.classification.prostate_x.processor import  ProstateXPatchProcessor
from fuse_examples.imaging.classification.prostate_x.post_processor import post_processing
from fuse.data.processor.processor_dicom_mri import DicomMRIProcessor


def process_mri_series():
    seq_to_use_dict = \
        {
            't2_tse_tra': 'T2',
            't2_tse_tra_Grappa3': 'T2',
            't2_tse_tra_320_p2': 'T2',

            'ep2d-advdiff-3Scan-high bvalue 100': 'b',
            'ep2d-advdiff-3Scan-high bvalue 500': 'b',
            'ep2d-advdiff-3Scan-high bvalue 1400': 'b',
            'ep2d_diff_tra2x2_Noise0_FS_DYNDISTCALC_BVAL': 'b',

            'ep2d_diff_tra_DYNDIST': 'b_mix',
            'ep2d_diff_tra_DYNDIST_MIX': 'b_mix',
            'diffusie-3Scan-4bval_fs': 'b_mix',
            'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen': 'b_mix',
            'diff tra b 50 500 800 WIP511b alle spoelen': 'b_mix',

            'ep2d_diff_tra_DYNDIST_MIX_ADC': 'ADC',
            'diffusie-3Scan-4bval_fs_ADC': 'ADC',
            'ep2d-advdiff-MDDW-12dir_spair_511b_ADC': 'ADC',
            'ep2d-advdiff-3Scan-4bval_spair_511b_ADC': 'ADC',
            'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC': 'ADC',
            'diff tra b 50 500 800 WIP511b alle spoelen_ADC': 'ADC',
            'ADC_S3_1': 'ADC',
            'ep2d_diff_tra_DYNDIST_ADC': 'ADC',

        }

    # patients with special fix
    exp_patients = ['ProstateX-0191', 'ProstateX-0148', 'ProstateX-0180']
    seq_to_use = ['T2', 'b', 'b_mix', 'ADC', 'ktrans']
    subseq_to_use = ['T2', 'b400', 'b800', 'ADC', 'ktrans']

    SER_INX_TO_USE = {}
    SER_INX_TO_USE['all'] = {'T2': -1, 'b': [0, 2], 'ADC': 0, 'ktrans': 0}
    SER_INX_TO_USE['ProstateX-0148'] = {'T2': 1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}
    SER_INX_TO_USE['ProstateX-0191'] = {'T2': -1, 'b': [0, 0], 'ADC': 0, 'ktrans': 0}
    SER_INX_TO_USE['ProstateX-0180'] = {'T2': -1, 'b': [1, 2], 'ADC': 0, 'ktrans': 0}

    # sequences with special fix
    B_SER_FIX = ['diffusie-3Scan-4bval_fs',
                 'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
                 'diff tra b 50 500 800 WIP511b alle spoelen']

    return seq_to_use_dict, SER_INX_TO_USE, exp_patients,seq_to_use,subseq_to_use

def prostate_x_dataset(paths,train_common_params,lgr):
    #### Train Data

    lgr.info(f'Train Data:', {'attrs': 'bold'})

    ## Create data source:
    DATABASE_REVISION = train_common_params['db_version']
    lgr.info(f'database_revision={DATABASE_REVISION}', {'color': 'magenta'})

    # create data source
    train_data_source = ProstateXDataSourcePatient(paths['data_dir'], 'train',
                                                       db_ver=train_common_params['db_version'],
                                                       db_name=train_common_params['db_name'],
                                                       fold_no=train_common_params['fold_no'])

    ## Create data processors:
    image_processing_args = {
        'patch_xy': 74,
        'patch_z': 13,
    }

    ## Create data processor

    seq_to_use_dict, SER_INX_TO_USE, \
    exp_patients, seq_to_use, subseq_to_use = process_mri_series()

    generate_processor = ProstateXPatchProcessor(
        vol_processor=DicomMRIProcessor(reference_inx=0,
                                            seq_dict=seq_to_use_dict,
                                            seq_to_use=seq_to_use,
                                            subseq_to_use=subseq_to_use,
                                            ser_inx_to_use=SER_INX_TO_USE,
                                            exp_patients=exp_patients),

        path_to_db=paths['data_dir'],
        data_path=paths['prostate_data_path'],
        ktrans_data_path=paths['ktrans_path'],
        db_name=train_common_params['db_name'],
        db_version=train_common_params['db_version'],
        fold_no=train_common_params['fold_no'],
        lsn_shape=(image_processing_args['patch_z'],
                   image_processing_args['patch_xy'],
                   image_processing_args['patch_xy']),
    )

    train_post_processor = partial(post_processing)

    # data augmentation (optional)
    num_channels = train_common_params['backbone_model_dict']['input_channels_num']
    slice_num = image_processing_args['patch_z']

    image_channels = [list(range(0, slice_num))]
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
            aug_op_affine,
            {'rotate': Uniform(-3.0, 3.0),
             'translate': (RandInt(-2, 2), RandInt(-2, 2)),
             'flip': (False, False),
             'scale': Uniform(0.9, 1.1),
             'channels': Choice(image_channels, probabilities=None)},
            {'apply': RandBool(0.5) if train_common_params['data.aug.phase_misalignment'] else 0}
        ],
        [
            ('data.input',),
            unsqueeze_2d_to_3d,
            {'channels': num_channels, 'axis_squeeze': 'z'},
            {}
        ],
    ]
    augmentor = AugmentorDefault(augmentation_pipeline=aug_pipeline)

    # Create dataset
    train_dataset = DatasetGenerator(cache_dest=paths['cache_dir'],
                                         data_source=train_data_source,
                                         processor=generate_processor,
                                         post_processing_func=train_post_processor,
                                         augmentor=augmentor,
                                         statistic_keys=['data.ground_truth']
                                         )

    gpu_ids_for_caching = []
    lgr.info(f'- Load and cache data:')

    train_dataset.create()

    lgr.info(f'- Load and cache data: Done')

    #### Validation data
    lgr.info(f'Validation Data:', {'attrs': 'bold'})

    ## Create data source
    validation_data_source = ProstateXDataSourcePatient(paths['data_dir'], 'validation',
                                                            db_ver=DATABASE_REVISION,
                                                            db_name=train_common_params['db_name'],
                                                            fold_no=train_common_params['fold_no'])

    # post processor
    validation_post_processor = partial(post_processing)

    ## Create dataset
    validation_dataset = DatasetGenerator(cache_dest=paths['cache_dir'],
                                              data_source=validation_data_source,
                                              processor=generate_processor,
                                              post_processing_func=validation_post_processor,
                                              augmentor=None,
                                              statistic_keys=['data.ground_truth']
                                              )

    lgr.info(f'- Load and cache data:')

    validation_dataset.create(num_workers=0)
    lgr.info(f'Data - task caching and filtering:', {'attrs': 'bold'})

    return train_dataset, validation_dataset