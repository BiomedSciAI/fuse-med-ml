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
    loss = -(target * logprobs).sum() / target.sum()  # logits.shape[0]
    return loss


def supervised_contrastive_loss(
    feat_xi: Tensor,
    feat_xj: Tensor,
    yi: Tensor,
    yj: Tensor,
    symm: bool = False,
    alpha: float = 0.5,
    temp: float = 0.2,
) -> Tensor:
    """
    :param feat_i:               feature in the shape [Batch_size, embed dim]
    :param feat_j:               feature in the shape [Batch_size, embed dim]
    :param target_i:             ground truth label of class indexes in the shape [batch_size]
    :param target_j:             ground truth label of class indexes in the shape [batch_size]
    :param tempreture:           tempreture scaling parameter
    :param symmetrical:          should the output of the function is a combination of supervised contrastive loss components
    :param alpha:                if symmetrical -> the weight for each component
    """
    if feat_xi.isnan().any() | feat_xj.isnan().any():
        raise Exception("One of the features has nan value")

    feat_xi = F.normalize(feat_xi, p=2, dim=1)
    feat_xj = F.normalize(feat_xj, p=2, dim=1)
    label_vec = torch.unsqueeze(yi, 0)
    mask = torch.eq(torch.transpose(label_vec, 0, 1), yj).float()
    if mask.sum() == 0:
        return torch.tensor(0.0).to(mask.device)
    logits_xi = torch.div(torch.matmul(feat_xi, torch.transpose(feat_xj, 0, 1)), temp)
    logits_max, _ = torch.max(logits_xi, dim=1, keepdim=True)
    logits_xi = logits_xi - logits_max.detach()
    loss_xi = softcrossentropyloss(mask, logits_xi)  # / torch.sum(mask, 1)
    if symm:
        logits_xj = torch.matmul(feat_xj, torch.transpose(feat_xi, 0, 1)) / temp
        loss_xj = softcrossentropyloss(mask, logits_xj)
        return alpha * loss_xi.sum() + (1 - alpha) * loss_xj.sum()
    else:
        return loss_xi.mean()


class SupervisedContrastiveLoss(LossBase):
    """
    Supervised contrastive loss as defined in 'https://arxiv.org/pdf/2004.11362.pdf'

    """

    def __init__(
        self,
        *,  # prevent positional args
        feat_i: str = None,
        feat_j: str = None,
        target_i: str = None,
        target_j: str = None,
        tempreture: float = 0.2,
        alpha: float = 0.5,
        symmetrical: bool = False,
        callable: Callable = None,
        weight: Optional[float] = None,
        preprocess_func: Optional[Callable] = None,
    ) -> None:
        """
        This class wraps a PyTorch loss function with a Fuse api.
        Args:
        :param feat_i:               batch_dict key for feat i (e.g., network feature output)
        :param feat_j:               batch_dict key for feat_j (e.g., other network feature output to contrast with)
        :param target_i:             batch_dict key for target (e.g., ground truth label for feat_i)
        :param target_j:             batch_dict key for target (e.g., ground truth label for feat_j)
        :param tempreture:           tempreture scaling parameter
        :param symmetrical:          should the output of the function is a combination of supervised contrastive loss components
        :param alpha:                if symmetrical -> the weight for each component

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

        self.temp = tempreture
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
        feat_j = batch_dict[self.feat_j]

        target_i = batch_dict[self.target_i]
        target_j = batch_dict[self.target_j]

        loss_obj = supervised_contrastive_loss(
            feat_i,
            feat_j,
            target_i,
            target_j,
            symm=self.symm,
            alpha=self.alpha,
            temp=self.temp,
        )
        if self.weight is not None:
            loss_obj *= self.weight

        return loss_obj
