import torch

from fuse.utils import NDict


class LossExtractFromBatchDict(torch.nn.Module):
    """
    Read the loss from batch dict
    Useful in cases that the model is the one that set the loss
    """

    def __init__(self, key: str) -> None:
        """
        Where to read the loss from
        """
        super().__init__()
        self.key = key

    def forward(self, batch_dict: NDict) -> torch.Tensor:
        return batch_dict[self.key]
