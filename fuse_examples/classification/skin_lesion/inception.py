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

try:
    from torch.hub import load_state_dict_from_url
except:
    pass

import logging
from typing import Optional
from fuse.models.backbones.backbone_inception_resnet_v2 import FuseBackboneInceptionResnetV2


class InceptionResnetV2(FuseBackboneInceptionResnetV2):
    """
    2D ResNet backbone
    """

    def __init__(self, pretrained_weights_url: Optional[str] = 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
                 input_channels_num: int = 3) -> None:
        """
        Create InceptionResnetV2
        :param pretrained_weights_url: url containing pretrained weights, if None pretrained weights won't be loaded
        :param input_channels_num:     number of channels in the input data
        """

        # init original model
        super().__init__(logical_units_num=43, input_channels_num=3)

        # load pretrained parameters if required
        if pretrained_weights_url is not None:
            try:
                state_dict = load_state_dict_from_url(pretrained_weights_url)
                self.load_state_dict(state_dict, strict=False)
            except AttributeError:
                logger = logging.getLogger('Fuse')
                logger.info('Invalid URL for InceptionResnetV2 pretrained weights')

        # save input parameters
        self.input_channels_num = input_channels_num

