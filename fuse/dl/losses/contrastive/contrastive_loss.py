from typing import Callable, Optional

from fuse.dl.losses.loss_base import LossBase
from fuse.utils.ndict import NDict

import torch
import torch.nn.functional as F
from torch import Tensor


def softcrossentropyloss(target: Tensor, logits: Tensor) -> Tensor:
    """
    From the pytorch discussion Forum:
    https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
    """
    logprobs = F.log_softmax(logits, dim=1)
    # print(target * logprobs)
    loss = -(target * logprobs).sum() / target.sum()  # logits.shape[0]
    return loss


def supervised_contrastive_loss(
    feat_xi: Tensor, feat_xj: Tensor, yi: Tensor, yj: Tensor, symm: bool = False, alpha: float = 0.5, temp: float = 0.2
) -> Tensor:
    if feat_xi.isnan().any() | feat_xj.isnan().any():
        print("feature has nan")
    if len(feat_xi.shape) < 2:
        feat_xi = feat_xi.unsqueeze(dim=0)
    if len(feat_xj.shape) < 2:
        feat_xj = feat_xj.unsqueeze(dim=0)
    feat_xi = F.normalize(feat_xi, p=2, dim=1)
    feat_xj = F.normalize(feat_xj, p=2, dim=1)
    label_vec = torch.unsqueeze(yi, 0)
    mask = torch.eq(torch.transpose(label_vec, 0, 1), yj).float()
    if mask.sum() == 0:
        return torch.tensor(0.0).to("cuda")
    logits_xi = torch.div(torch.matmul(feat_xi, torch.transpose(feat_xj, 0, 1)), temp)
    logits_max, _ = torch.max(logits_xi, dim=1, keepdim=True)
    logits_xi = logits_xi - logits_max.detach()
    loss_xi = softcrossentropyloss(mask, logits_xi)  # / torch.sum(mask, 1)
    if symm:
        logits_xj = torch.matmul(feat_xj, torch.transpose(feat_xi, 0, 1)) / temp
        loss_xj = softcrossentropyloss(mask, logits_xj) / torch.sum(mask, 0)
        return alpha * loss_xi.sum() + (1 - alpha) * loss_xj.sum()
    else:
        return loss_xi.mean()


class SupervisedContrastiveLoss(LossBase):
    """
    Default Fuse loss function

    '''

    """

    def __init__(
        self,
        *,  # prevent positional args
        feat_i: str = None,
        feat_j: str = None,
        target_i: str = None,
        target_j: str = None,
        temp: float = 0.2,
        alpha: float = 0.5,
        symmetrical: bool = False,
        callable: Callable = None,
        weight: Optional[float] = None,
        preprocess_func: Optional[Callable] = None,
    ) -> None:
        """
        This class wraps a PyTorch loss function with a Fuse api.
        Args:
        :param pred:               batch_dict key for prediction (e.g., network output)
        :param target:             batch_dict key for target (e.g., ground truth label)
        :param callable:           PyTorch loss function handle (e.g., torch.nn.functional.cross_entropy)
        :param weight:             scalar loss multiplier
        :param preprocess_func:             function that filters batch_dict/ The function gets an input batch_dict and returns filtered batch_dict
            the expected function signature is:
                foo(batch_dict: NDict) -> NDict:
        """
        super().__init__()
        self.feat_i = feat_i
        self.feat_j = feat_j
        self.target_i = target_i
        self.target_j = target_j

        self.temp = temp
        self.alpha = alpha
        self.symm = symmetrical

        self.callable = callable
        self.weight = weight
        self.preprocess_func = preprocess_func

    def forward(self, batch_dict: NDict) -> torch.Tensor:
        # preprocess batch_dict if required
        if self.preprocess_func is not None:
            batch_dict = self.preprocess_func(batch_dict)
        feat_i = batch_dict[self.feat_i]
        feat_j = batch_dict[self.feat_i]

        target_i = batch_dict[self.target_i]
        target_j = batch_dict[self.target_j]

        loss_obj = supervised_contrastive_loss(
            feat_i, feat_j, target_i, target_j, symm=self.symm, alpha=self.alpha, temp=self.temp
        )
        if self.weight is not None:
            loss_obj *= self.weight

        return loss_obj
