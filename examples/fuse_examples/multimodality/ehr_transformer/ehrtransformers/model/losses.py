import warnings

from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional
from torch.nn.modules.loss import _Loss

# from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    BSD 2-Clause "Simplified" License
    
    The loss function SupConLoss in losses.py takes features (L2 normalized) and labels as input, and return the loss. If labels is None or not passed to the loss, it degenerates to SimCLR.
    """
    def __init__(self, n_features=2, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        """

        :param n_features: number of features that are to be similar per entity
        :param temperature:
        :param contrast_mode:
        :param base_temperature:
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.n_features = n_features
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        labels = None
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        if len(features.shape) == 2:
            rem = features.shape[0]%self.n_features
            if rem == 0:
                features = features.reshape(features.shape[0]//self.n_features, -1, features.shape[1])
            else:
                #This means the minibatch is not divisible by n_features (i.e. it does not contain whole groups
                # of similar entities). It cannot happen during training, when a minibatch is generated so that every
                # entity is duplicated n_features times, but it can happen during validation/test, when the original
                # order of a set is preserved
                if features.shape[0] > rem:
                    features = features[:-rem,:]
                    features = features.reshape(features.shape[0] // self.n_features, -1, features.shape[1])
                else:
                    return 0.0
                # raise ValueError('`batch size must be divisible by number of instances of the same entity (n_features) used for contrastive loss')
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        features = F.normalize(features, dim=2, p=2)



        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastivePositiveLoss(_Loss):
    r"""Creates a criterion that measures the loss given input tensors that should be similar
    :math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    The loss function for each sample is:

    .. math::
        \text{loss}(x, y) =
        \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y = 1 \\
        \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
        \end{cases}

    Args:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)`, where `N` is the batch size and `D` is the embedding dimension.
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as Input1.
        - Target: :math:`(N)` or :math:`()`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`, otherwise scalar.
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float = 0., size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ContrastivePositiveLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)
