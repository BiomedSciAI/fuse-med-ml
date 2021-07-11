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

from typing import Sequence, Dict, Tuple

import torch
import torch.nn as nn


from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
import numpy as np
from fuse_examples.classification.prostate_x.head_1d_classifier import FuseHead1dClassifier


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
                 ch_num: None = None,
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
            self.conv = conv3x3(5, 16)
        else:
            self.conv = conv3x3(len(self.ch_num), 16)


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
        self.max_pool = nn.MaxPool3d((2,2,2),stride=(2,2,2))
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

    def forward(self,batch_dict):
        if len(self.conv_inputs)>1:
            tensors_list = [FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]).float() for conv_input in self.conv_inputs]
            max_tensor_dim = max([len(tmp_tensor.shape) for tmp_tensor in tensors_list])
            conv_input = torch.cat([tmp_tensor.unsqueeze_(0) if len(tmp_tensor.shape)<max_tensor_dim else tmp_tensor for tmp_tensor in tensors_list], 1)
        else:
            conv_input = torch.cat([FuseUtilsHierarchicalDict.get(batch_dict, conv_input[0]) for conv_input in self.conv_inputs], 1)

        if (self.ch_num is not None):
            if (conv_input.shape[0]>len(self.ch_num)):
                conv_input = conv_input[:,self.ch_num, :, :, :]


        if len(conv_input.shape)<4:
            conv_input = conv_input.unsqueeze_(0)

        out = self.conv(conv_input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.max_pool(out)
        out = self.layer2(out)
        out = self.max_pool(out)
        out = self.layer3(out)
        out = self.max_pool(out)
        out = self.layer4(out)
        out = self.max_pool(out)
        out = self.layer5(out)
        out = self.max_pool(out)

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
    Sequence:
        1. Starts with low resolution image to extract low resolution features and attention score per each image patch
        2. Extract high resolution features 'k' most significant patches (using the attention score)
        3. Use low resolution features, high resolution features and attention score to classify the image
    Input: high resolution tensor [BATCH_SIZE, 1, H, W]
    Output:
        TBD
    """

    def __init__(self,
                 conv_inputs: Tuple[Tuple[str, int], ...] = (('data.input', 1),),
                 backbone: ResNet = ResNet(),
                 heads: Sequence[torch.nn.Module] = (FuseHead1dClassifier(),),
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
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features', features)
        FuseUtilsHierarchicalDict.set(batch_dict, 'model.backbone_features', features)

        for head in self.heads:
            batch_dict = head.forward(batch_dict)

        return batch_dict['model']

if __name__ == '__main__':

    import torch
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    DEVICE = 'cpu'  # 'cuda'
    DATAPARALLEL = False  # True
    num_features = 515
    model = Fuse_model_3d_multichannel(
        conv_inputs=(('data.input', 1),),
        backbone= ResNet(),
        heads=[
        FuseHead1dClassifier(head_name='ClinSig',
                                        conv_inputs=[('model.backbone_features', num_features)],
                                        post_concat_inputs=None,
                                        dropout_rate=0.25,
                                        shared_classifier_head=None,
                                        layers_description=None,
                                        num_classes=2),

        ]
    )

    model = model.to(DEVICE)
    if DATAPARALLEL:
        model = torch.nn.DataParallel(model)

    import gzip
    import pickle

    real_data_example = '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/fus_sessions/prostate/ProstateX_runs/V2/cache_prostate_x_size=74x74x13_spacing=0.5x0.5X3_fold0/0000000001.pkl.gz'
    with gzip.open(real_data_example, 'rb') as pickle_file:
        dummy_data = pickle.load(pickle_file)
        clinical_data = torch.rand([1]).to(DEVICE)
        clinical_data = clinical_data.unsqueeze(0)
        input = FuseUtilsHierarchicalDict.get(dummy_data, 'data.input')
        FuseUtilsHierarchicalDict.set(dummy_data,'data.input',input)
        zone = FuseUtilsHierarchicalDict.get(dummy_data, 'data.zone')
        zone2feature = {
                        'PZ':torch.tensor(np.array([0,0,0]),dtype=torch.float32).unsqueeze(0),
                        'TZ': torch.tensor(np.array([0,0,1]), dtype=torch.float32).unsqueeze(0),
                        'AS': torch.tensor(np.array([0,1,0]), dtype=torch.float32).unsqueeze(0),
                        'SV': torch.tensor(np.array([1,0,0]), dtype=torch.float32).unsqueeze(0),
                        }
        FuseUtilsHierarchicalDict.set(dummy_data, 'data.input', torch.cat((input.unsqueeze(0),input.unsqueeze(0)),dim=0))
        FuseUtilsHierarchicalDict.set(dummy_data, 'data.tensor_clinical', torch.cat((zone2feature[zone],zone2feature[zone]),dim=0))


    res = {}
    # manager = FuseManagerDefault()
    # manager.set_objects(net=model)
    # checkpoint =  '/gpfs/haifa/projects/m/msieve_dev3/usr/Tal/fus_sessions/multiclass_MG/malignant_multi_class_sentara_baptist_froedtert/exp_12_pretrain_normal-mal-benign_head/model_2/checkpoint_80_epoch.pth'
    # aa = manager.load_checkpoint(checkpoint)
    res['model'] = model.forward(dummy_data)
    print('Forward pass shape - head_0: ', end='')
    print(str(res['model']['logits']['head_1'].shape))

    print('\nForward pass shape - head_1: ', end='')
    print(str(res['model']['logits']['head_1'].shape))

    total_params = sum(p.numel() for p in model.parameters())
    if not DATAPARALLEL:
        backbone_params = sum(p.numel() for p in model._modules['backbone_clinical'].parameters())
        print('backbone_clinical params = %d' % backbone_params)
        print('Heads params = %d' % (total_params - backbone_params))

    print('\nTotal params = %d' % total_params)
