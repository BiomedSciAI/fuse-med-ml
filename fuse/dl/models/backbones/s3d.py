from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fuse.dl.models.backbones.backbone_resnet import BackboneResnet
# from torchvision.models import resnet18, resnet34, resnet50

class Model3DContext(nn.Module):
    """
    Capture not only local feature about a slice, but also information about the 3D context
    """

    def __init__(
        self,
        in_features: int,
        first_kernel: List[Tuple[int, int, int]] = [(3, 3, 3), (2, 1, 1)],
        second_kernel: List[Tuple[int, int, int]] = [(3, 1, 1), (2, 1, 1)],
        expansion: int = 1,
    ) -> None:
        """
        Create simple 3D context model
        :param in_features: number of input features
        :param first_kernel: Tuple for kernel and Tuple for stride (3D)
        :param second_kernel: Tuple for kernel and Tuple for stride (3D)
        :param features expansion factor for each conv layer
        """
        super().__init__()
        self.expansion = expansion

        # activation
        self.relu = nn.ReLU()

        out_features = in_features
        self.conv3d_1 = nn.Conv3d(
            out_features,
            self.expansion * out_features,
            kernel_size=first_kernel[0],
            stride=first_kernel[1],
            padding=(1, 1, 1),
        )
        out_features *= self.expansion
        self.bn3d_1 = nn.BatchNorm3d(out_features)
        self.conv3d_2 = nn.Conv3d(
            out_features,
            self.expansion * out_features,
            kernel_size=second_kernel[0],
            stride=second_kernel[1],
            padding=(2, 0, 0),
        )
        out_features *= self.expansion
        self.bn3d_2 = nn.BatchNorm3d(out_features)
        self.conv_pooling_3d = nn.Sequential(
            self.conv3d_1, self.bn3d_1, self.relu, self.conv3d_2, self.bn3d_2, self.relu
        )

        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """
        Forward pass
        :param x: input tensor representing spatial features with 3D context. shape: [batch_size, in_features, z, y, x]
        :return: spatial features with 3D context. shape: [batch_size, out_features, z', y, x]
        """
        return self.conv_pooling_3d(x)  # type: ignore

    def get_out_features(self) -> int:
        """
        :return: the number of output_features
        """
        return self.out_features

class ModelSliced3D(nn.Module):
    """
    3D model base on 2D backbone applied on each slice, followed by the 3D convolution layers and 3D classifier.
    """
        
    def __init__(self, backbone_2d: nn.Module, context_3d: nn.Module, pool: bool = False) -> None:
        """
        Create the model
        :param backbone_2d: backbone model
        :param context_3d: model which captures the 3D context
        """
        super().__init__()

        # store input parameters
        self.backbone_2d = backbone_2d
        self.context_3d = context_3d
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self._pool = pool
        # auxiliary classifier - currently not helping
        # self.conv_classifier_2d = nn.Sequential(
        #    nn.AdaptiveMaxPool3d(output_size=(None, 1, 1)),
        #    nn.Dropout3d(p=classifier_dropout),
        #    nn.Conv3d(n_features, self.num_classes + 1, kernel_size=1),
        # )

    def features(self, x: Tensor) -> Any:
        """
        Extract spatial features - given a 3D tensor
        :param x: Input tensor - shape: [batch_size, channels, z, y, x]
        :return: spatial features - shape [batch_size, n_features, z', y', x']
        """
        # x assumed to by
        batch_size = x.shape[0]
        x = x.permute((0, 2, 1, 3, 4))  # permute to [batch_size, z, channels, y, x]
        x = x.reshape(
            (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        )  # reshaped to [batch_size * z, channels, y, x]

        x = self.backbone_2d(x)

        x = x.reshape(
            (batch_size, x.shape[0] // batch_size, x.shape[1], x.shape[2], x.shape[3])
        )  # reshaped to [batch_size, z, channels', x', y']
        x = x.permute(
            (0, 2, 1, 3, 4)
        )  # reshaped to [batch_size, n_mid_features, z, y', x']

        # conv pooling
        if self.context_3d is not None:
            x = self.context_3d(x)  # shape: [batch_size, n_features, z', y', x']

        return x  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. 3D global classification given a volume
        :param x: Input volume. shape: [batch_size, channels, z, y, x]
        :return: Tensor of features
        """
        x = self.features(x)  # type: ignore
        if self._pool:
            x = self.avgpool(x)
            x = x.flatten(1)
        return x

