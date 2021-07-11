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

from typing import Union, Sequence

import numpy as np
import torch


def calculate_box_iou(box1: Union[np.ndarray, torch.Tensor],
                      box2: Union[np.ndarray, torch.Tensor],
                      api: str = 'np') -> np.ndarray:
    """
    Calculates intersection-over-union for two sets of bounding boxes, for all possible pairs.
    This method supports both NumPy and PyTorch tensor inputs - remember to specify in the 'api' param.
    :param box1: list of bounding boxes, shape [N, 4+] where first four entries in each row are top-left-bottom-right coordinates.
                 rows may contain more than four entries, only the first four are used.
    :param box2: same as box1 (e.g., use box1 for preds and box2 for targets)
    :param api:  select either 'np' or 'torch'
    :return: A NumPy matrix of IOU for all pair combinations of box1/box2. rows=box1, cols=box2
    """

    assert api in ['np', 'torch']

    if api == 'np':
        n = int(box1.shape[0])
        m = int(box2.shape[0])
        lt = np.maximum(
            np.repeat(box1[:, :2][:, None, :], m, axis=1),  # [n, 2] --> [n, m, 2]
            np.repeat(box2[:, :2][None, :, :], n, axis=0)  # [m, 2] --> [n, m, 2]
        )
        rb = np.minimum(
            np.repeat(box1[:, 2:4][:, None, :], m, axis=1),
            np.repeat(box2[:, 2:4][None, :, :], n, axis=0)
        )
        wh = (rb - lt).clip(min=0)  # [N,M,2]

    elif api == 'torch':
        n = box1.size(0)
        m = box2.size(0)
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(n, m, 2),  # [n, 2] -> [n, 1, 2] -> [n, m, 2]
            box2[:, :2].unsqueeze(0).expand(n, m, 2),  # [m ,2] -> [1, m, 2] -> [n, m, 2]
        )
        rb = torch.min(
            box1[:, 2:4].unsqueeze(1).expand(n, m, 2),
            box2[:, 2:4].unsqueeze(0).expand(n, m, 2),
        )
        wh = (rb - lt).clamp(min=0)  # [N,M,2]

    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

    if api == 'np':
        area1 = np.repeat(area1[:, None], m, axis=1)
        area2 = np.repeat(area2[None, :], n, axis=0)
    elif api == 'torch':
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)

    if api == 'torch':
        iou = iou.cpu().numpy()

    return iou


def box_non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, overlap_threshold: float = 0.5) -> Sequence:
    """
    Bounding box non-maximum-suppression.
    Suppresses boxes with intersection-over-union above overlap_threshold, taking the top-scoring overlap box according
    to 'scores' param.

    Adapted and modified from:
    https://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    :param boxes:               ndarray, shaped [num_boxes, 4], entries in each row are top-left-bottom-right coordinates.
    :param scores:              ndarray, shaped [num_boxes, ]
    :param overlap_threshold:   threshold for IoU suppression

    :return:  list of indices of non-suppressed boxes
    """

    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute areas
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort by scores, descending order
    idxs = np.argsort(scores)[::-1]

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap >= overlap_threshold)[0])))

    return pick


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    visualize = True

    preds = [[100, 250, 400, 450, 0, 0.9],
             [300, 700, 700, 900, 0, 0.8],
             [600, 400, 1000, 1000, 0, 0.4]]
    targets = [[150, 250, 500, 500, 0],
               [200, 710, 880, 950, 0],
               [530, 320, 920, 1100, 0]]

    preds = np.array(preds)
    targets = np.array(targets)
    ious = calculate_box_iou(preds, targets, api='np')
    print('np')
    print(ious)
    ious = calculate_box_iou(torch.Tensor(preds), torch.Tensor(targets), api='torch')
    print('torch')
    print(ious)

    # Visualize data
    if visualize:
        fig, ax = plt.subplots(1)
        ax.imshow(np.zeros([1200, 1200]), cmap='gray')
        for idx, bb in enumerate(preds):
            rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            text_val = ', '.join(["%.2f" % score for score in ious[idx] if score > 0])
            ax.text(bb[0] + 10, bb[1] - 10, text_val, color='red')
        for idx, bb in enumerate(targets):
            rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    # Test non maximum suppression
    all_boxes = np.array([box[:4] for box in preds] + [box[:4] for box in targets]).astype('int')
    all_scores = np.array([0.9, 0.8, 0.4, 1.0, 1.0, 1.0])

    nms_indices = box_non_maximum_suppression(all_boxes, all_scores, overlap_threshold=0.5)
    print('\nAfter non-maximum-suppression')
    print(all_boxes[nms_indices])
    print(all_scores[nms_indices])

