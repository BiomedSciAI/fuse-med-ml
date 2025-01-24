import torch.nn.functional as F
from monai.networks.nets import UNet as UNetBase
from torch import nn

from fuse.utils.ndict import NDict


class UNet(nn.Module):
    def __init__(self, input_name: str, seg_name: str, unet_kwargs: dict):
        super().__init__()
        self.input_name = input_name
        self.seg_name = seg_name
        self.unet = UNetBase(**unet_kwargs)

    def forward(self, batch_dict: NDict) -> None:
        x = batch_dict[self.input_name]
        seg_output = self.unet(x)
        batch_dict[self.seg_name] = F.softmax(seg_output, dim=1)
        return batch_dict
