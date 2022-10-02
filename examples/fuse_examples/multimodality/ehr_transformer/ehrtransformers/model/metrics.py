from sklearn import metrics
from torch import nn
import numpy as np
import torch

def get_nontrivial_inds(label):
    """

    :param label: tensor of binary GT labels, that contains only '1.0' and '0.0'
    :return: indices of columns that contain both positive and negative classes
    """
    return [ind for ind in range(label.shape[1]) if ((1.0 in label[:,ind]) and (0.0 in label[:,ind]))]

def precision(logits, label):
    if isinstance(label, list):
        label = torch.from_numpy(np.vstack(label))
    if isinstance(logits, list):
        logits = torch.from_numpy(np.vstack(logits))
    inds = get_nontrivial_inds(label)
    logits = logits[:, inds]
    label = label[:, inds]
    sig = nn.Sigmoid()
    output = sig(logits)
    label, output = label.cpu(), output.detach().cpu()
    tempprc = metrics.average_precision_score(
        label.numpy(), output.numpy(), average="macro"
    )
    return tempprc, output, label


def auroc(logits, label):
    if isinstance(label, list):
        label = torch.from_numpy(np.vstack(label))
    if isinstance(logits, list):
        logits = torch.from_numpy(np.vstack(logits))
    inds = get_nontrivial_inds(label)
    logits = logits[:, inds]
    label = label[:, inds]

    sig = nn.Sigmoid()
    output = sig(logits)
    label, output = label.cpu(), output.detach().cpu()

    tempprc = metrics.roc_auc_score(label.numpy(), output.numpy(), average="macro")
    #     roc = metrics.roc_auc_score()
    return tempprc


def precision_test(logits, label):
    if isinstance(label, list):
        label = torch.from_numpy(np.vstack(label))
    if isinstance(logits, list):
        logits = torch.from_numpy(np.vstack(logits))

    inds = get_nontrivial_inds(label)
    logits = logits[:, inds]
    label = label[:, inds]

    sig = nn.Sigmoid()
    output = sig(logits)
    tempprc = metrics.average_precision_score(
        label.numpy(), output.numpy(), average="macro"
    )
    #     roc = metrics.roc_auc_score()
    return tempprc, output, label


def auroc_test(logits, label):
    if isinstance(label, list):
        label = torch.from_numpy(np.vstack(label))
    if isinstance(logits, list):
        logits = torch.from_numpy(np.vstack(logits))

    inds = get_nontrivial_inds(label)
    logits = logits[:, inds]
    label = label[:, inds]

    sig = nn.Sigmoid()
    output = sig(logits)
    tempprc = metrics.roc_auc_score(label.numpy(), output.numpy(), average="macro")
    #     roc = metrics.roc_auc_score()
    return tempprc
