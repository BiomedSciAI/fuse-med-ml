import torch
import torch.nn as nn
import torch.nn.functional as F
from fuse.models.backbones.backbone_mlp import FuseMultilayerPerceptronBackbone
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from typing import Dict, Tuple, Sequence
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2



# class Fusefchead(nn.Module):
#     def __init__(self,
#                  cat_representations: Sequence[Tuple[str, int]] = (('model.cat_representations', 1),),
#                  backbone: FuseMultilayerPerceptronBackbone = FuseMultilayerPerceptronBackbone(
#                      layers=[2], mlp_input_size=512),
#                  ) -> None:
#         super().__init__()
#
#         self.cat_representations = cat_representations
#         self.backbone = backbone
#     def forward(self, batch_dict: Dict) -> Dict:
#         cat_representations = FuseUtilsHierarchicalDict.get(batch_dict, self.cat_representations[0][0])
#         logits = self.backbone(cat_representations)
#         preds = F.softmax(logits, dim=1)
#
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.logits', logits)
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.output', preds)
#
#         return batch_dict

# class FuseModelImagingTabularHead(torch.nn.Module):
#     def __init__(self,
#                  backbone: torch.nn.Module,
#                  heads: Sequence[torch.nn.Module],
#                  ) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.heads = torch.nn.ModuleList(heads)
#         self.add_module('heads', self.heads)
#
#     def forward(self, batch_dict: Dict) -> Dict:
#         representations_batch_dict = self.backbone.forward(batch_dict)
#         imaging_representations = FuseUtilsHierarchicalDict.get(representations_batch_dict, 'imaging_representations')
#         tabular_representations = FuseUtilsHierarchicalDict.get(representations_batch_dict, 'tabular_representations')
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_representations', imaging_representations)
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.tabular_representations', tabular_representations)
#         if len(imaging_representations.shape)<2:
#             imaging_representations = imaging_representations.unsqueeze(dim=0)
#         if len(tabular_representations.shape)<2:
#             tabular_representations = tabular_representations.unsqueeze(dim=0)
#         cat_representations = torch.cat((tabular_representations, imaging_representations), 1)
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.cat_representations', cat_representations)
#         for head in self.heads:
#             batch_dict = head.forward(batch_dict)
#         return batch_dict['model']

# class Fusesoftmax(nn.Module):
#     def __init__(self,
#                  logits: Sequence[Tuple[str, int]] = (('model.features', 1),),
#                  ) -> None:
#         super().__init__()
#
#         self.logits = logits
#     def forward(self, batch_dict: Dict) -> Dict:
#         logits = FuseUtilsHierarchicalDict.get(batch_dict, self.logits[0][0])
#         preds = F.softmax(logits, dim=1)
#
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.logits', logits)
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.output', preds)
#
#         return batch_dict

# class FuseModelTabularImaging(torch.nn.Module):
#     def __init__(self,
#                  continuous_tabular_input: Tuple[Tuple[str, int], ...],
#                  categorical_tabular_input: Tuple[Tuple[str, int], ...],
#                  imaging_inputs: Tuple[Tuple[str, int], ...],
#                  backbone_categorical_tabular: torch.nn.Module = None,
#                  backbone_continuous_tabular: torch.nn.Module = None,
#                  backbone_imaging: torch.nn.Module = None,
#                  projection_imaging: nn.Conv2d = nn.Conv2d(384, 256, kernel_size=1, stride=1)
#                  ) -> None:
#         super().__init__()
#         self.continuous_tabular_input = continuous_tabular_input
#         self.categorical_tabular_input = categorical_tabular_input
#         self.imaging_inputs = imaging_inputs
#         self.backbone_categorical_tabular = backbone_categorical_tabular
#         self.backbone_continuous_tabular = backbone_continuous_tabular
#         self.backbone_imaging = backbone_imaging
#         self.projection_imaging = projection_imaging
#
#     def forward(self, batch_dict: Dict) -> Dict:
#
#         #tabular encoder
#         categorical_input = FuseUtilsHierarchicalDict.get(batch_dict, self.categorical_tabular_input[0][0])
#         categorical_embeddings = self.backbone_categorical_tabular(categorical_input)
#         continuous_input = FuseUtilsHierarchicalDict.get(batch_dict, self.continuous_tabular_input[0][0])
#         input_cat = torch.cat((categorical_embeddings, continuous_input), 1)
#         tabular_representations = self.backbone_continuous_tabular(input_cat) #dim 256
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.tabular_representations', tabular_representations)
#
#         #imaging encoder
#         imaging_input = FuseUtilsHierarchicalDict.get(batch_dict, self.imaging_inputs[0][0])
#         backbone_imaging_features = self.backbone_imaging.forward(imaging_input)
#         res = F.max_pool2d(backbone_imaging_features, kernel_size=backbone_imaging_features.shape[2:])
#         imaging_representations = self.projection_imaging.forward(res)
#         imaging_representations = torch.squeeze(imaging_representations)
#         FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_representations', imaging_representations)
#
#         return batch_dict['model']



# concat model
class TabularImagingConcat(nn.Module):
    def __init__(self, pooling='max',projection_imaging: nn.Conv2d = nn.Conv2d(384, 256, kernel_size=1, stride=1)):
        super().__init__()
        assert pooling in ('max', 'avg')
        self.pooling = pooling
        self.projection_imaging = projection_imaging

    def fix_imaging(self,imaging_features):
        if self.pooling == 'max':
            imaging_features = F.max_pool2d(imaging_features, kernel_size=imaging_features.shape[2:])

        elif self.pooling == 'avg':
            imaging_features = F.avg_pool2d(imaging_features, kernel_size=imaging_features.shape[2:])

        imaging_features = self.projection_imaging.forward(imaging_features)
        imaging_features = torch.squeeze(torch.squeeze(imaging_features,dim=3),dim=2)
        return imaging_features


    def forward(self, batch_dict):

        imaging_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.imaging_features')
        tabular_features = FuseUtilsHierarchicalDict.get(batch_dict, 'model.tabular_features')
        imaging_features = self.fix_imaging(imaging_features)
        res = torch.cat([tabular_features, imaging_features], dim=1)
        return res


#Tabular model
class FuseModelTabularContinuousCategorical(torch.nn.Module):
    def __init__(self,
                 continuous_tabular_input: Tuple[Tuple[str, int], ...],
                 categorical_tabular_input: Tuple[Tuple[str, int], ...],
                 backbone_categorical_tabular: FuseMultilayerPerceptronBackbone,
                 backbone_continuous_tabular: FuseMultilayerPerceptronBackbone,
                 heads: Sequence[torch.nn.Module],
                 ) -> None:
        super().__init__()
        self.continuous_tabular_input = continuous_tabular_input
        self.categorical_tabular_input = categorical_tabular_input
        self.backbone_categorical_tabular = backbone_categorical_tabular
        self.backbone_cat_tabular = backbone_continuous_tabular
        # self.add_module('backbone', self.backbone)
        self.heads = torch.nn.ModuleList(heads)
        self.add_module('heads', self.heads)

    def forward(self, batch_dict: Dict) -> Dict:
        categorical_input = FuseUtilsHierarchicalDict.get(batch_dict, self.categorical_tabular_input[0][0])
        categorical_embeddings = self.backbone_categorical_tabular(categorical_input)
        continuous_input = FuseUtilsHierarchicalDict.get(batch_dict, self.continuous_tabular_input[0][0])
        input_cat = torch.cat((categorical_embeddings, continuous_input), 1)
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
                 multimodal_backbone: torch.nn.Module=None,
                 tabular_heads: Sequence[torch.nn.Module]=None,
                 imaging_heads: Sequence[torch.nn.Module]=None,
                 multimodal_heads: Sequence[torch.nn.Module]=None,
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


        self.tabular_heads = torch.nn.ModuleList(tabular_heads)
        if self.tabular_heads:
            self.add_module('tabular_heads', self.tabular_heads)

        self.imaging_heads = torch.nn.ModuleList(imaging_heads)
        if self.imaging_heads:
            self.add_module('imaging_heads', self.imaging_heads)

        self.multimodal_heads = torch.nn.ModuleList(multimodal_heads)
        if self.multimodal_heads:
            self.add_module('multimodal_heads', self.multimodal_heads)

    def tabular_modules(self):
        return [self.tabular_backbone, self.tabular_heads]

    def imaging_modules(self):
        return [self.imaging_backbone, self.imaging_heads]

    def multimodal_modules(self):
        return [self.multimodal_backbone, self.multimodal_heads]

    def forward(self, batch_dict: Dict) -> Dict:

        if self.tabular_backbone:
            tabular_features = self.tabular_backbone.forward(batch_dict)

        if self.imaging_backbone:
            imaging_input = FuseUtilsHierarchicalDict.get(batch_dict, self.imaging_inputs[0][0])
            imaging_features = self.imaging_backbone.forward(imaging_input)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.imaging_features', imaging_features)

        if self.multimodal_backbone:
            multimodal_features = self.multimodal_backbone.forward(batch_dict)
            FuseUtilsHierarchicalDict.set(batch_dict, 'model.multimodal_features', multimodal_features)


        # run through heads
        if self.tabular_heads:
            for head in self.tabular_heads:
                batch_dict = head.forward(batch_dict)

        if self.imaging_heads:
            for head in self.imaging_heads:
                batch_dict = head.forward(batch_dict)

        if self.multimodal_heads:
            for head in self.multimodal_heads:
                batch_dict = head.forward(batch_dict)

        return batch_dict['model']


if __name__ == '__main__':
    import torch

    batch_dict = {'data.continuous': torch.randn(8, 14),
                  'data.categorical': torch.randn(8, 63),
                  'data.image': torch.randn(8, 1, 2200, 1200)}

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
    model_imaging = FuseBackboneInceptionResnetV2(input_channels_num=1)
    model_multimodel = TabularImagingConcat()

    model = FuseMultiModalityModel(
        tabular_inputs=(('data.continuous', 1),('data.categorical', 1),),
        imaging_inputs=(('data.image', 1),),
        tabular_backbone=model_tabular,
        imaging_backbone=model_imaging,
        multimodal_backbone = model_multimodel,
    )
    res = model.forward(batch_dict)
    a=1