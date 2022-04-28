import torch.optim as optim
from fuse_examples.classification.multimodality.model_tabular_imaging import *
from fuse.losses.loss_default import FuseLossDefault
import torch.nn.functional as F
from fuse.metrics.classification.metric_auc import FuseMetricAUC
from fuse.metrics.classification.metric_accuracy import FuseMetricAccuracy
from fuse_examples.classification.multimodality.loss_multimodal_contrastive_learning import FuseLossMultimodalContrastiveLearning
from fuse.models.model_default import FuseModelDefault
from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier
from fuse.models.heads.head_1d_classifier import FuseHead1dClassifier
from fuse.models.model_ensemble import FuseModelEnsemble

def multimodal_parameters(train_common_params: dict,infer_common_params: dict,analyze_common_params: dict):


    ################################################
    # backbone_models
    model_tabular = FuseModelTabularContinuousCategorical(
        continuous_tabular_input=(('data.continuous', 1),),
        categorical_tabular_input=(('data.categorical', 1),),
        backbone_categorical_tabular=train_common_params['tabular_encoder_categorical'],
        backbone_continuous_tabular = train_common_params['tabular_encoder_continuous'],
        heads=None,
        )

    model_imaging = train_common_params['imaging_encoder']
    model_multimodel_concat = TabularImagingConcat()
    heads_for_multimodal = {
                         'multimodal_head':
                             [
                                FuseHead1dClassifier(
                                    head_name='multimodal',
                                    conv_inputs=(('model.multimodal_features',
                                                  train_common_params['tabular_feature_size'] * 2),),
                                    num_classes=2,
                                )
                            ],
                         'tabular_head':
                             [
                                 FuseHead1dClassifier(
                                     head_name='tabular',
                                     conv_inputs=(('model.tabular_features',
                                                   train_common_params['tabular_feature_size']),),
                                     num_classes=2,
                                 )
                             ],
                         'imaging_head':
                             [
                                FuseHeadGlobalPoolingClassifier(
                                    head_name='imaging',
                                    dropout_rate=0.5,
                                    layers_description=(256,),
                                    conv_inputs=(('model.imaging_features',
                                                  train_common_params['imaging_feature_size']),),
                                    num_classes=2,
                                    pooling="avg",
                                )
                            ],


                        }
    loss_for_multimodal = {
                        'multimodal_loss':FuseLossDefault(pred_name='model.logits.multimodal', target_name='data.gt',
                                                                        callable=F.cross_entropy, weight=1.0),
                        'tabular_loss':FuseLossDefault(pred_name='model.logits.tabular', target_name='data.gt',
                                                                        callable=F.cross_entropy, weight=1.0),
                        'imaging_loss':FuseLossDefault(pred_name='model.logits.imaging', target_name='data.gt',
                                                                        callable=F.cross_entropy, weight=1.0),
                        'ensemble_loss':FuseLossDefault(pred_name='model.output.tabular_ensemble_average', target_name='data.gt',
                                                                        callable=F.nll_loss, weight=1.0,reduction='sum'),

                        }
    metric_for_multimodal = {
                        'multimodal_auc': FuseMetricAUC(pred_name='model.output.multimodal', target_name='data.gt'),
                        'tabular_auc': FuseMetricAUC(pred_name='model.output.tabular', target_name='data.gt'),
                        'imaging_auc': FuseMetricAUC(pred_name='model.output.imaging', target_name='data.gt'),
                        'ensemble_auc':FuseMetricAUC(pred_name='model.output.tabular_ensemble_average', target_name='data.gt'),
                        }
    ################################################


    if train_common_params['fusion_type'] == 'mono_tabular':

        train_common_params['model'] = FuseMultiModalityModel(
                                            tabular_inputs=(('data.continuous', 1), ('data.categorical', 1),),
                                            tabular_backbone=model_tabular,
                                            tabular_heads=heads_for_multimodal['tabular_head'],
                                        )
        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['tabular_loss'],
        }
        train_common_params['metrics'] = {
            'auc': metric_for_multimodal['tabular_auc'],
        }

        train_common_params['manager.learning_rate'] = 1e-4
        train_common_params['manager.weight_decay'] = 1e-4
        train_common_params['manager.momentum'] = 0.9
        train_common_params['manager.step_size'] = 150
        train_common_params['manager.gamma'] = 0.1
        train_common_params['optimizer'] = optim.SGD(train_common_params['model'].parameters(),
                  lr=train_common_params['manager.learning_rate'],
                  momentum=train_common_params['manager.momentum'],
                  weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.StepLR(train_common_params['optimizer'], step_size=train_common_params['manager.step_size'],
                                  gamma=train_common_params['manager.gamma'])

    if train_common_params['fusion_type'] == 'mono_imaging':

        train_common_params['model'] = FuseMultiModalityModel(
                                            imaging_inputs=(('data.image', 1),),
                                            imaging_backbone=model_imaging,
                                            imaging_heads=heads_for_multimodal['imaging_head'],
                                        )
        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['imaging_loss'],
        }
        train_common_params['metrics'] = {
            'auc': metric_for_multimodal['imaging_auc'],
        }
        train_common_params['manager.learning_rate'] = 1e-5
        train_common_params['manager.weight_decay'] = 0.001

        train_common_params['optimizer'] = optim.Adam(train_common_params['model'].parameters(), lr=train_common_params['manager.learning_rate'],
                                                         weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(train_common_params['optimizer'])

    if train_common_params['fusion_type'] == 'late_fusion':

        train_common_params['model'] = FuseMultiModalityModel(
                                            tabular_inputs=(('data.continuous', 1), ('data.categorical', 1),),
                                            imaging_inputs=(('data.image', 1),),
                                            tabular_backbone=model_tabular,
                                            imaging_backbone=model_imaging,
                                            multimodal_backbone=model_multimodel_concat,
                                            imaging_heads=heads_for_multimodal['imaging_head'],
                                            tabular_heads=heads_for_multimodal['tabular_head'],
                                            multimodal_heads=heads_for_multimodal['multimodal_head'],
                                        )

        train_common_params['loss'] = {
            'cls_loss': loss_for_multimodal['multimodal_loss'],
        }
        train_common_params['metrics'] = {
            'auc': metric_for_multimodal['multimodal_auc'],
        }
        train_common_params['manager.learning_rate'] = 1e-4
        train_common_params['manager.weight_decay'] = 1e-4
        train_common_params['manager.momentum'] = 0.9
        train_common_params['manager.step_size'] = 150
        train_common_params['manager.gamma'] = 0.1
        train_common_params['optimizer'] = optim.SGD(train_common_params['model'].parameters(),
                  lr=train_common_params['manager.learning_rate'],
                  momentum=train_common_params['manager.momentum'],
                  weight_decay=train_common_params['manager.weight_decay'])
        train_common_params['scheduler'] = optim.lr_scheduler.StepLR(train_common_params['optimizer'], step_size=train_common_params['manager.step_size'],
                                  gamma=train_common_params['manager.gamma'])

    if train_common_params['fusion_type'] == 'ensemble':

        train_common_params['tabular_dir'] = '/projects/msieve_dev3/usr/Tal/my_research/multi-modality/model_mg_radiologist_usa/mono_tabular/'
        train_common_params['imaging_dir'] = '/projects/msieve_dev3/usr/Tal/my_research/multi-modality/model_mg_radiologist_usa/mono_imaging_no_aug/'
        train_common_params['model'] = FuseModelEnsemble(input_model_dirs=[train_common_params['tabular_dir'],
                                                                           train_common_params['imaging_dir']])
        infer_common_params['model_dir'] = [train_common_params['tabular_dir'],
                                                                           train_common_params['imaging_dir']]

        infer_common_params['output_keys'] = ['data.gt',
                                              'model.output.ensemble_output_0.tabular',
                                              'model.output.ensemble_output_1.tabular',
                                              'model.output.ensemble_output_1.imaging',
                                              'model.output.tabular_ensemble_average',
                                              'model.output.tabular_ensemble_majority_vote']

        analyze_common_params['metrics'] =  train_common_params['metrics'] = {
                                                    'auc': metric_for_multimodal['ensemble_auc'],
                                                }


        # train_common_params['loss'] = {
        #     'cls_loss': loss_for_multimodal['ensemble_loss'],
        #
        # }
        # train_common_params['metrics'] = {
        #     'auc': metric_for_multimodal['ensemble_auc'],
        # }
        # train_common_params['manager.learning_rate'] = 1e-5
        # train_common_params['manager.weight_decay'] = 0.001
        #
        # train_common_params['optimizer'] = optim.Adam(train_common_params['model'].parameters(), lr=train_common_params['manager.learning_rate'],
        #                                                  weight_decay=train_common_params['manager.weight_decay'])
        # train_common_params['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(train_common_params['optimizer'])


    #Mo:different parameter
    if train_common_params['fusion_type'] == 'cotrastive':
        train_common_params['model'] = FuseModelTabularImaging(

            continuous_tabular_input=(('data.continuous', 1),),
            categorical_tabular_input=(('data.categorical', 1),),
            imaging_inputs=(('data.image', 1),),
            backbone_categorical_tabular=train_common_params['tabular_encoder_categorical'],
            backbone_continuous_tabular=train_common_params['tabular_encoder_continuous'],
            backbone_imaging=train_common_params['imaging_encoder'],

        )

        train_common_params['loss'] = FuseLossMultimodalContrastiveLearning(
            imaging_representations='model.imaging_representations',
            tabular_representations='model.tabular_representations',
            label='data.gt',
            temperature=0.1,
            alpha=0.5)
        train_common_params['metrics'] = None

    return train_common_params,infer_common_params,analyze_common_params