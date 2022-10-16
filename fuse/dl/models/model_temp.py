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

from cmath import log
from typing import Sequence, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fuse.utils.ndict import NDict
import numpy as np



class LinearLayers(nn.Module):
    def __init__(self,
                 in_ch: Sequence[int] = (256,),
                 layers_description: Sequence[int] = (256,),
                 dropout_rate: float = 0.1,):

        super(LinearLayers, self).__init__()
        layer_list = []
        layer_list.append(nn.Linear(in_ch, layers_description[0]))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        if len(layers_description)>1:
            for curr_layer_size in layers_description[1:]:
                layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
                layer_list.append(nn.ReLU())
                if dropout_rate is not None and dropout_rate > 0:
                    layer_list.append(nn.Dropout(p=dropout_rate))
                last_layer_size = curr_layer_size

        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x

class Head1DClassifier(nn.Module):
    def __init__(self,
                 head_name: str = 'head_0',
                 conv_inputs: Sequence[Tuple[str, int]] = None, 
                 num_classes: int = 2,
                 post_concat_inputs: Optional[Sequence[Tuple[str, int]]] = None,
                 post_concat_model: Optional[Sequence[int]] = None,
                 layers_description: Sequence[int] = (256,),
                 dropout_rate: float = 0.1,
                 shared_classifier_head: Optional[torch.nn.Module] = None,

                 ) -> None:
        """
        Classifier head 1d.

        Output of a forward pass:
        'model.logits.head_name' and 'outputs.head_name', both in shape [batch_size, num_classes]

        :param head_name:                   batch_dict key
        :param conv_inputs:                 List of feature map inputs - tuples of (batch_dict key, channel depth)
                                            If multiple inputs are used, they are concatenated on the channel axis
                for example:
                conv_inputs=(('model.backbone_features', 193),)
        :param num_classes:                 Number of output classes (per feature map location)
        :param post_concat_inputs:          Additional vector (one dimensional) inputs, concatenated just before the classifier module
        :param post_concat_model            Layers description for the post_concat_inputs module - sequence of hidden layers sizes
        :param layers_description:          Layers description for the classifier module - sequence of hidden layers sizes
        :param dropout_rate:                Dropout rate for classifier module layers
        :param shared_classifier_head:      Optional reference for external torch.nn.Module classifier
        """
        super().__init__()


        self.head_name = head_name
        assert conv_inputs is not None, 'conv_inputs must be provided'
        self.conv_inputs = conv_inputs
        self.post_concat_inputs = post_concat_inputs
        self.post_concat_model = post_concat_model

        feature_depth = sum([conv_input[1] for conv_input in self.conv_inputs])

        if (post_concat_inputs is not None) and (self.post_concat_model is None):
            ## concat post_concat_input directly to conv_input,
            feature_depth += sum([post_concat_input[1] for post_concat_input in post_concat_inputs])

        elif self.post_concat_model is not None:
            # concat post_concat_input features from classifier_post_concat_model to conv_input
            features_depth_post_concat = sum([post_concat_input[1] for post_concat_input in post_concat_inputs])
            self.classifier_post_concat_model = LinearLayers(in_ch=features_depth_post_concat,
                                                           layers_description=self.post_concat_model,
                                                           dropout_rate=dropout_rate)




        if shared_classifier_head is not None:
            self.classifier_head_module = shared_classifier_head
        else:
            self.classifier_head_module = ClassifierLinear(in_ch=feature_depth,
                                                        num_classes=num_classes,
                                                        layers_description=layers_description,
                                                        dropout_rate=dropout_rate)




    def forward(self,batch_dict: Dict) -> Dict:

        conv_input = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs])

        res = conv_input


        if self.post_concat_inputs is not None:
            post_concat_input = torch.cat(
                [batch_dict[post_concat_input[0]] for post_concat_input in self.post_concat_inputs])
            if self.post_concat_model is None:
                # concat post_concat_input directly to conv_input
                res = torch.cat([res, post_concat_input],dim=1)
            else:
                # concat post_concat_input features from classifier_post_concat_model to conv_input
                post_concat_input_feat = self.classifier_post_concat_model(post_concat_input)
                res = torch.cat([res, post_concat_input_feat], dim=1)

        logits = self.classifier_head_module(res)  # --> res.shape = [batch_size, 2, 1, 1]
        if len(logits.shape) > 2:
            logits = logits.squeeze(dim=3)  # --> res.shape = [batch_size, 2, 1]
            logits = logits.squeeze(dim=2)  # --> res.shape = [batch_size, 2]

        cls_preds = F.softmax(logits, dim=1)

        batch_dict['model.logits.' + self.head_name] = logits
        batch_dict['model.output.' + self.head_name] = cls_preds

        return batch_dict

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)



# Residual block
class ResidualBlock(torch.nn.Module):


    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample



    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        res_mod = 'concat'
        if res_mod=='sum':
            if self.downsample:
                residual = self.downsample(x)
            out += residual
        elif res_mod=='concat':
            out = torch.cat((residual,out),1)
        out = self.relu(out)
        return out


# ResNet
class ResNet(torch.nn.Module):
    def __init__(self,
                 conv_inputs:Tuple[Tuple[str, int], ...] = (('data.input', 1),),
                 ch_num: int = None,
                 ) -> None:

        super(ResNet, self).__init__()
        block = ResidualBlock
        self.ch_num = ch_num
        layers = [1,1,1,1,1]
        out_features = [32,64,128,256,512]
        self.in_channels = 16
        in_features = [self.in_channels ,48,112,240,496]
        self.conv_inputs = conv_inputs
        if self.ch_num is None:
            self.conv = conv3x3(1, 16)
        else:
            self.conv = conv3x3(self.ch_num, 16)


        self.bn = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, out_features[0], layers[0])
        self.in_channels = in_features[1]
        self.layer2 = self.make_layer(block, out_features[1], layers[1])
        self.in_channels = in_features[2]
        self.layer3 = self.make_layer(block, out_features[2], layers[2])
        self.in_channels = in_features[3]
        self.layer4 = self.make_layer(block, out_features[3], layers[3])
        self.in_channels = in_features[4]
        self.layer5 = self.make_layer(block, out_features[4], layers[4])
        self.max_pool2 = nn.MaxPool3d((2,2,2),stride=(2,2,2))
        self.max_pool1 = nn.MaxPool3d((1,2,2),stride=(1,2,2))

        self.conv_last = conv3x3(1008, 1008)
        self.bn_last = nn.BatchNorm3d(1008)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1008, 2048)
        self.fc2 = nn.Linear(2048, 512)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm3d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, batch_dict: NDict):
        if len(self.conv_inputs)>1:
            tensors_list = [batch_dict[conv_input[0]].float() for conv_input in self.conv_inputs]
            max_tensor_dim = max([len(tmp_tensor.shape) for tmp_tensor in tensors_list])
            conv_input = torch.cat([tmp_tensor.unsqueeze_(0) if len(tmp_tensor.shape)<max_tensor_dim else tmp_tensor for tmp_tensor in tensors_list], 1)
        else:
            conv_input = torch.cat([batch_dict[conv_input[0]] for conv_input in self.conv_inputs], 1)


        if len(conv_input.shape)<4:
            conv_input = conv_input.unsqueeze_(0)

        out = self.conv(conv_input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool2(out)

        out = self.layer1(out)
        out = self.max_pool2(out)
        out = self.layer2(out)
        out = self.max_pool2(out)
        out = self.layer3(out)
        out = self.max_pool1(out)
        out = self.layer4(out)
        out = self.max_pool1(out)
        out = self.layer5(out)
        out = self.max_pool1(out)

        out = self.conv_last(out)
        out = self.bn_last(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.dropout(out)

        return out


class Fuse_model_3d_multichannel(torch.nn.Module):
    """
    Fuse model that classifing high resolution images

    """

    def __init__(self,
                 conv_inputs: Tuple[Tuple[str, int], ...] = (('data.input', 1),),
                 backbone: ResNet = None,
                 heads: Sequence[torch.nn.Module] = None,
                 ch_num = None,
                 ) -> None:
        """
        Default Fuse model - convolutional neural network with multiple heads
        :param conv_inputs:     batch_dict name for model input and its number of input channels
        :param backbone:        PyTorch backbone module - a convolutional neural network
        :param heads:           Sequence of head modules
        """
        super().__init__()

        self.conv_inputs = conv_inputs
        if backbone is None:
            backbone = ResNet(conv_inputs=conv_inputs)
        if heads is None:
            heads =  (Head1DClassifier(),)
        self.backbone = backbone
        self.heads = torch.nn.ModuleList(heads)
        self.add_module('heads', self.heads)
        self.ch_num = ch_num



    def forward(self,
                batch_dict: Dict) -> Dict:
        """
        Forward function of the model
        :param input: Tensor [BATCH_SIZE, 1, H, W]
        :return: classification scores - [BATCH_SIZE, num_classes]
        """

        features = self.backbone(batch_dict)
        batch_dict['model.backbone_features'] = features

        for head in self.heads:
            batch_dict = head.forward(batch_dict)
        
        return batch_dict


class ClassifierLinear(nn.Module):
    def __init__(self,
                 in_ch: Sequence[int] = (256,),
                 num_classes: float = 2,
                 layers_description: Sequence[int] = (256,),
                 dropout_rate: float = 0.1,):

        super(ClassifierLinear, self).__init__()

        layer_list = []
        if layers_description is not None:
            layer_list.append(nn.Linear(in_ch, layers_description[0]))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = layers_description[0]
            if len(layers_description)>1:
                for curr_layer_size in layers_description[1:]:
                    layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
                    layer_list.append(nn.ReLU())
                    if dropout_rate is not None and dropout_rate > 0:
                        layer_list.append(nn.Dropout(p=dropout_rate))
                    last_layer_size = curr_layer_size
        else:
            last_layer_size = in_ch

        layer_list.append(nn.Linear(last_layer_size, num_classes))
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x

