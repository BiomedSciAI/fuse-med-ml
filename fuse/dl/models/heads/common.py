"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class ClassifierFCN(nn.Module):
    """
    Sequence of (Conv2D 1X1 , ReLU, Dropout). The length of the sequence and layers size defined by layers_description
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: Optional[int],
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
    ):
        """
        :param in_ch: Number of input channels
        :param num_classes: Appends Conv2D(last_layer_size, num_classes, kernel_size=1, stride=1) if num_classes is not None
        :param layers_description: defines the length of the sequence and layers size.
        :param dropout_rate: if 0 will not include the dropout layers
        """
        super().__init__()
        layer_list = []
        last_layer_size = in_ch

        for i in range(len(layers_description)):
            curr_layer_size = layers_description[i]
            layer_list.append(
                nn.Conv2d(last_layer_size, curr_layer_size, kernel_size=1, stride=1)
            )
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(
                nn.Conv2d(last_layer_size, num_classes, kernel_size=1, stride=1)
            )

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x


class ClassifierFCN3D(nn.Module):
    """
    Sequence of (Conv3D 1X1 , ReLU, Dropout). The length of the sequence and layers size defined by layers_description
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: Optional[int],
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
    ):
        """
        :param in_ch: Number of input channels
        :param num_classes: Appends Conv2D(last_layer_size, num_classes, kernel_size=1, stride=1) if num_classes is not None
        :param layers_description: defines the length of the sequence and layers size.
        :param dropout_rate: if 0 will not include the dropout layers
        """
        super().__init__()
        layer_list = []
        last_layer_size = in_ch

        for curr_layer_size in layers_description:
            layer_list.append(
                nn.Conv3d(last_layer_size, curr_layer_size, kernel_size=1, stride=1)
            )
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(
                nn.Conv3d(last_layer_size, num_classes, kernel_size=1, stride=1)
            )

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x


class ClassifierMLP(nn.Module):
    """
    Sequence of (Linear , ReLU, Dropout). The length of the sequence and layers size defined by layers_description
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: Optional[int] = None,
        layers_description: Sequence[int] = (256,),
        dropout_rate: float = 0.1,
        bias: bool = True,
    ):
        """
        :param in_ch: Number of input channels
        :param num_classes: Appends Conv2D(last_layer_size, num_classes, kernel_size=1, stride=1) if num_classes is not None
        :param layers_description: defines the length of the sequence and layers size.
        :param dropout_rate: if 0 will not include the dropout layers
        """
        super().__init__()
        layer_list = []
        last_layer_size = in_ch
        for curr_layer_size in layers_description:
            layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size

        if num_classes is not None:
            layer_list.append(
                nn.Linear(
                    in_features=last_layer_size, out_features=num_classes, bias=bias
                )
            )

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x


class EncoderEmbeddingOutputHead(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        layers: List[int],
        dropout: float,
        num_classes: int,
        pooling: str = None,
    ):
        """
        NOTE: This is work in progress. Do not use for now.

        This class applies a multi-layer MLP to an input and allows to apply a pooling operation to the sequence dimension - prior to applying the MLP.
        This is usefull for extracting a single representation for embeddings of an entire sequence.
        Args:
            embedding_size: MLP input dimension.
            layers: List[int], specifies the output dimension of the MLP in each layer.
            dropout: dropout rate, applied to every layer in the MLP
            pooling: str (optional) type of pooling to be used, currently available are ["mean", "last"]. Pooling operations ignore pad tokens - a padding mask should be supplied in the forward pass.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.layers = layers
        self.dropout = dropout
        self.pooling_type = pooling

        # this weird assignment is for backward compatability
        # when loading pretrained weights
        self.classifier = ClassifierMLP(
            in_ch=embedding_size,
            layers_description=layers,
            dropout_rate=dropout,
            num_classes=num_classes,
        ).classifier

        if pooling is not None:
            self.pooling = ModularPooling1D(pooling=pooling)
        else:
            self.pooling = None

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor = None,
        keep_pool_dim: bool = True,
    ) -> Tensor:
        """
        Args:
            padding_mask: a mask that indicates which positions are for valid tokens (1) and which are padding tokens (0) - typically this is similar to an attention mask.
            keep_pool_dim: if True an output of shape (B, L, D) will be returned as (B, 1, D) otherwise returns (B, D)
        """

        if self.pooling is not None:
            assert (
                padding_mask is not None
            ), "OutputHead attempts to perform pooling - requires the padding_mask to detect padding tokens (usually same as the attention mask to the decoder), but padding_mask is None"

            inputs = self.pooling(
                inputs=inputs, padding_mask=padding_mask, keep_dim=keep_pool_dim
            )

        y = self.classifier(inputs)
        return y


class ModularPooling1D(nn.Module):
    """
    A wrapper around multiple pooling methods.
    Args:
        pooling: str, type of pooling to apply, available methods are: ["mean", "last"] TODO: add max?
        pool_dim: dimension to apply pooling
    """

    def __init__(self, pooling: str, pool_dim: int = 1, **kwargs: dict):
        super().__init__()

        self.pooling_type = pooling
        self.pool_dim = pool_dim

        if pooling in ["mean", "avg"]:  # pools the mean value of none-pad elements

            def _mean_pool(inputs: Tensor, last_valid_indices: Tensor) -> Tensor:
                inputs = inputs.cumsum(dim=self.pool_dim)
                outputs = self._extract_indices(
                    inputs, last_valid_indices, dim=self.pool_dim
                )
                outputs = outputs / (last_valid_indices + 1)
                return outputs

            self.pooling = lambda inputs, indices: _mean_pool(inputs, indices)

        elif pooling == "last":  # pools the last element that is not a PAD value

            def _last_pool(inputs: Tensor, last_valid_indices: Tensor) -> Tensor:
                return self._extract_indices(
                    inputs, last_valid_indices, dim=self.pool_dim
                )

            self.pooling = lambda inputs, indices: _last_pool(inputs, indices)

        else:
            raise NotImplementedError

    def _extract_indices(self, inputs: Tensor, indices: Tensor, dim: int = 1) -> Tensor:
        assert (
            dim == 1
        ), "extract indices for pooling head not implemented for dim != 1 yet"
        # extract indices in dimension using diffrentiable ops
        indices = indices.reshape(-1)
        index = indices.unsqueeze(1).unsqueeze(1)
        index = index.expand(size=(index.shape[0], 1, inputs.shape[-1]))
        pooled = torch.gather(inputs, dim=dim, index=index).squeeze(1)
        return pooled

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor = None,
        keep_dim: bool = True,
    ) -> Tensor:
        """
        See OutputHead().forward for a detailed description.
        """
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(torch.bool)
        # get indices of last positions of no-pad tokens
        last_valid_indices = get_last_non_pad_token(
            padding_mask=padding_mask
        ).unsqueeze(1)
        out = self.pooling(inputs, last_valid_indices)
        if keep_dim:
            out = out.unsqueeze(self.pool_dim)
        return out


def get_last_non_pad_token(padding_mask: Tensor) -> Tensor:
    """
    Returns the positions of last non-pad token, for every element in the batch.
    Expected input shape is (B, L), B is the batch size, L is the sequence dimension.
    Args:
        padding_mask: a boolean tensor with True values for none-padded positions and False values for padded positions (usually same as the attention mask input to an encoder model)
    """
    non_pad_pos = padding_mask.cumsum(dim=-1)  # starts from 1
    non_pad_last_pos = non_pad_pos[:, -1] - 1

    return non_pad_last_pos
