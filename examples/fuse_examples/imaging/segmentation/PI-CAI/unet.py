from torch import nn
from monai.networks.nets import UNet as UNetBase
from fuse.utils.ndict import NDict
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(
        self, input_name: str, seg_name: str, pre_softmax: str, post_softmax: str, out_features: int, unet_kwargs: dict
    ):
        super().__init__()
        self.input_name = input_name
        self.seg_name = seg_name
        self.pre_softmax = pre_softmax
        self.post_softmax = post_softmax

        self.unet = UNetBase(**unet_kwargs)
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.linear = nn.Linear(in_features=unet_kwargs["channels"][-1], out_features=out_features)
        # # extract bottom activation
        # self.activations = {}

        # def get_activation(name):
        #     def hook(model, input, output):
        #         self.activations[name] = output.detach()

        #     return hook

        # bottom_module = self.get_bottom_module()
        # # register last convolution module in the encoder
        # bottom_module.register_forward_hook(get_activation("bottom_layer"))

    def get_bottom_module(self):
        # specific code for the monai unet model, extract the last convolution of the encoder
        cur = self.unet.model
        for i in range(len(self.unet.strides)):
            cur = cur[1].submodule
        cur = cur.conv
        return cur

    def forward(self, batch_dict: NDict):
        x = batch_dict[self.input_name]
        seg_output = self.unet(x)
        # bottom_output = self.activations["bottom_layer"]
        batch_dict[self.seg_name] = torch.unsqueeze(seg_output[:,0,:,:,:], 1)
        # x = self.avgpool(bottom_output)
        # x = x.flatten(1)
        # batch_dict[self.pre_softmax] = x
        # x = self.linear(x)
        # batch_dict[self.post_softmax] = F.softmax(x, dim=1)
        return batch_dict
