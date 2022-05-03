import torch
import torch.nn as nn
import torch.nn.functional as F
from fuse.models.backbones.backbone_mlp import FuseMultilayerPerceptronBackbone
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from typing import Dict, Tuple, Sequence
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2

class project_imaging(nn.Module):

    def __init__(self, pooling='max', dim='2d', projection_imaging: nn.Module = None):
        super().__init__()
        assert pooling in ('max', 'avg')
        assert dim in ('2d', '3d')
        self.pooling = pooling
        self.dim = dim
        self.projection_imaging = projection_imaging

    def forward(self, imaging_features):
        if self.pooling == 'max':
            if self.dim == '2d':
                imaging_features = F.max_pool2d(imaging_features, kernel_size=imaging_features.shape[2:])
            else:
                imaging_features = F.max_pool3d(imaging_features, kernel_size=imaging_features.shape[2:])

        elif self.pooling == 'avg':
            if self.dim == '2d':
                imaging_features = F.avg_pool2d(imaging_features, kernel_size=imaging_features.shape[2:])
            else:
                imaging_features = F.max_pool3d(imaging_features, kernel_size=imaging_features.shape[2:])

        if self.projection_imaging is not None:
            imaging_features = self.projection_imaging.forward(imaging_features)
            imaging_features = torch.squeeze(torch.squeeze(imaging_features,dim=3),dim=2)

        return imaging_features


class project_tabular(nn.Module):

    def __init__(self, projection_tabular: nn.Module = None):
        super().__init__()
        self.projection_tabular = projection_tabular

    def forward(self, tabular_features):
        if self.projection_imaging is not None:
            tabular_features = self.projection_tabular.forward(tabular_features)
            tabular_features = torch.squeeze(torch.squeeze(tabular_features, dim=3), dim=2)

        return tabular_features

# concat model
class TabularImagingConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_dict):
        tabular_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.tabular_features')
        imaging_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.imaging_features')
        res = torch.cat([tabular_features, imaging_features], dim=1)
        return res

#Tabular model
class FuseModelTabularContinuousCategorical(torch.nn.Module):
    def __init__(self,
                 continuous_tabular_input: Tuple[Tuple[str, int], ...],
                 categorical_tabular_input: Tuple[Tuple[str, int], ...],
                 backbone_categorical_tabular: torch.nn.Module,
                 backbone_continuous_tabular: torch.nn.Module,
                 backbone_cat_tabular: torch.nn.Module,
                 heads: Sequence[torch.nn.Module],
                 ) -> None:
        super().__init__()
        self.continuous_tabular_input = continuous_tabular_input
        self.categorical_tabular_input = categorical_tabular_input
        self.backbone_categorical_tabular = backbone_categorical_tabular
        self.backbone_continuous_tabular = backbone_continuous_tabular
        self.backbone_cat_tabular = backbone_cat_tabular
        self.heads = torch.nn.ModuleList(heads)
        self.add_module('heads', self.heads)

    def forward(self, batch_dict: Dict) -> Dict:
        if self.backbone_categorical_tabular:
            categorical_input = FuseUtilsHierarchicalDict.get(batch_dict, self.categorical_tabular_input[0][0])
            categorical_embeddings = self.backbone_categorical_tabular(categorical_input)
        else:
            categorical_embeddings = FuseUtilsHierarchicalDict.get(batch_dict, self.categorical_tabular_input[0][0])

        if self.backbone_continuous_tabular:
            continuous_input = FuseUtilsHierarchicalDict.get(batch_dict, self.continuous_tabular_input[0][0])
            continuous_embeddings = self.backbone_categorical_tabular(continuous_input)
        else:
            continuous_embeddings = FuseUtilsHierarchicalDict.get(batch_dict, self.continuous_tabular_input[0][0])

        input_cat = torch.cat((categorical_embeddings, continuous_embeddings), 1)
        tabular_features = self.backbone_cat_tabular(input_cat)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.tabular_features', tabular_features)

        for head in self.heads:
            batch_dict = head.forward(batch_dict)
        return batch_dict['model']

#Tabular Imaging model
class FuseMultiModalityModel(torch.nn.Module):
    def __init__(self,
                 tabular_inputs: Tuple[Tuple[str, int], ...]=None,
                 imaging_inputs: Tuple[Tuple[str, int], ...]=None,
                 tabular_backbone: torch.nn.Module=None,
                 imaging_backbone: torch.nn.Module=None,
                 tabular_projection: torch.nn.Module=None,
                 imaging_projection: torch.nn.Module = None,
                 multimodal_backbone: torch.nn.Module=None,
                 heads: Sequence[torch.nn.Module]=None,
                 ) -> None:
        super().__init__()

        self.tabular_inputs = tabular_inputs
        self.tabular_backbone = tabular_backbone
        if self.tabular_backbone:
            self.add_module('tabular_backbone', self.tabular_backbone)

        self.imaging_inputs = imaging_inputs
        self.imaging_backbone = imaging_backbone
        if self.imaging_backbone:
            self.add_module('imaging_backbone', self.imaging_backbone)

        self.multimodal_backbone = multimodal_backbone
        if self.multimodal_backbone:
            self.add_module('multimodal_backbone', multimodal_backbone)

        self.tabular_projection = tabular_projection
        if self.tabular_projection:
            self.add_module('tabular_projection', tabular_projection)

        self.imaging_projection = imaging_projection
        if self.imaging_projection:
            self.add_module('imaging_projection', imaging_projection)

        self.heads = torch.nn.ModuleList(heads)
        if self.heads:
            self.add_module('heads', self.heads)


    def forward(self, batch_dict: Dict) -> Dict:

        if self.tabular_backbone:
            tabular_features = self.tabular_backbone.forward(batch_dict)

        if self.imaging_backbone:
            imaging_input = FuseUtilsHierarchicalDict.get(batch_dict, self.imaging_inputs[0][0])
            imaging_features = self.imaging_backbone.forward(imaging_input)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_features', imaging_features)

        if self.tabular_projection:
            tabular_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.tabular_features')
            tabular_features = self.tabular_projection.forward(tabular_features)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.tabular_features', tabular_features)

        if self.imaging_projection:
            imaging_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.imaging_features')
            imaging_features = self.imaging_projection.forward(imaging_features)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_features', imaging_features)

        if self.multimodal_backbone:
            multimodal_features = self.multimodal_backbone.forward(batch_dict)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.multimodal_features', multimodal_features)


        # run through heads
        if self.heads:
            for head in self.heads:
                batch_dict = head.forward(batch_dict)

        return batch_dict['model']


if __name__ == '__main__':
    import torch
    from fuse.models.heads.head_1d_classifier import FuseHead1dClassifier
    from fuse.models.heads.head_global_pooling_classifier import FuseHeadGlobalPoolingClassifier

    batch_dict = {'data.continuous': torch.randn(8, 14),
                  'data.categorical': torch.randn(8, 63),
                  'data.image': torch.randn(8, 1, 2200, 1200)}


    heads_for_multimodal = {
                         'multimodal_head':
                             [
                                FuseHead1dClassifier(
                                    head_name='multimodal',
                                    conv_inputs=(('model.multimodal_features',
                                                  256 * 2),),
                                    num_classes=2,
                                )
                            ],
                         'tabular_head':
                             [
                                 FuseHead1dClassifier(
                                     head_name='tabular',
                                     conv_inputs=(('model.tabular_features',
                                                   256),),
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
                                                  384),),
                                    num_classes=2,
                                    pooling="avg",
                                )
                            ],


                        }


    # model = FuseModelTabularImaging(
    #     continuous_tabular_input=(('data.continuous', 1),),
    #     categorical_tabular_input=(('data.categorical', 1),),
    #     imaging_inputs=(('data.patch.input.input_0', 1),),)
    #
    # res = model(batch_dict)

    # model = FuseModelTabularContinuousCategorical(
    #     continuous_tabular_input=(('data.continuous', 1),),
    #     categorical_tabular_input=(('data.categorical', 1),),
    #     backbone_categorical_tabular=FuseMultilayerPerceptronBackbone(layers=[128, 63],mlp_input_size=63),
    #     backbone_continuous_tabular = FuseMultilayerPerceptronBackbone( layers=[256],mlp_input_size=77),
    #     heads=None,
    #     )
    # res = model.forward(batch_dict)

    model_tabular = FuseModelTabularContinuousCategorical(
        continuous_tabular_input=(('data.continuous', 1),),
        categorical_tabular_input=(('data.categorical', 1),),
        backbone_categorical_tabular=FuseMultilayerPerceptronBackbone(layers=[128, 63],mlp_input_size=63),
        backbone_continuous_tabular = FuseMultilayerPerceptronBackbone( layers=[256],mlp_input_size=77),
        heads=None,
        )
    model_projector_imaging = project_imaging(projection_imaging=nn.Conv2d(384, 256, kernel_size=1, stride=1))
    model_projector_tabular = None
    model_imaging = FuseBackboneInceptionResnetV2(input_channels_num=1)
    model_multimodel = TabularImagingConcat()

    model = FuseMultiModalityModel(
        tabular_inputs=(('data.continuous', 1),('data.categorical', 1),),
        imaging_inputs=(('data.image', 1),),
        tabular_backbone=model_tabular,
        imaging_backbone=model_imaging,
        tabular_projection=model_projector_tabular,
        imaging_projection=model_projector_imaging,
        multimodal_backbone = model_multimodel,
        heads=[heads_for_multimodal['multimodal_head'][0]],
    )
    res = model.forward(batch_dict)
    a=1