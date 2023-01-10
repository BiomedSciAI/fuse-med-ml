from typing import Dict, Optional

import numpy as np

from scipy.spatial import distance
from scipy.spatial.distance import directed_hausdorff


class MetricsSegmentation:
    @staticmethod
    def get_tf_ft_values_from_bool_array_with_weights(u: np.ndarray, v: np.ndarray, w: Optional[Dict[int, np.ndarray]]):
        """
        Calculates the false positive and true negative between two ndarray , while having different weight for each pixel
        :param u: single sample prediction matrix ( np.ndarray of any shape) per sample
        :param v: target mask ( np.ndarray of any shape ) per sample
        :param w: Optional dictionary - key = label ,value = weight per pixel . Each element is  float in range [0-1]
        """
        not_u = 1.0 - u
        not_v = 1.0 - v
        not_u = w * not_u
        u = w * u
        nft = (not_u * v).sum()
        ntf = (u * not_v).sum()
        return (nft, ntf)

    @staticmethod
    def dice(pred: np.ndarray, target: np.ndarray, pixel_weight: Optional[Dict[int, np.ndarray]] = None) -> Dict:
        """
        Compute dice similarity score (2*|X&Y| / (|X|+|Y|)) using sklearn , pred and target should be of same shape
        Supports multiclass (semantic segmentation)
        :param pred: single sample prediction matrix ( np.ndarray of any shape of type int/bool ) per sample
        :param target: target mask ( np.ndarray of any shape of type int/bool) per sample
        :param pixel_weight: Optional dictionary - key = label ,value = weight per pixel . Each element is  float in range [0-1]
        :return dice score
        """

        labels = np.unique(target)
        labels = labels[labels != 0]
        scores = {}
        for label in labels:
            mask_pred = pred == label
            mask_gt = target == label
            label = str(int(label))
            gt_empty = np.sum(mask_gt) == 0
            pred_empty = np.sum(mask_pred) == 0

            # dice not defined if both are empty ( 0/0 situation)
            if gt_empty and pred_empty:
                scores[label] = 1.0
            else:
                if pixel_weight is None or pixel_weight[label] is None:
                    scores[label] = 1.0 - distance.dice(mask_pred.flatten(), mask_gt.flatten())
                else:
                    scores[label] = 1.0 - distance.dice(
                        mask_pred.flatten(), mask_gt.flatten(), pixel_weight[label].flatten()
                    )

        return scores

    @staticmethod
    def iou_jaccard(pred: np.ndarray, target: np.ndarray, pixel_weight: Optional[Dict[int, np.ndarray]] = None) -> Dict:
        """
        Calculates intersection over union (iou) (|X&Y| / | XUY |) score based on predicted and target segmentation mask
        Supports multiclass (semantic segmentation)
        :param pred: single sample prediction matrix ( np.ndarray of any shape of type int/bool) per sample
        :param target: target mask ( np.ndarray of any shape of type int/bool) per sample
        :param pixel_weight: Optional dictionary - key = label ,value = weight per pixel . Each element is  float in range [0-1]
        :return: iou score
        """
        # extract info for the plot
        labels = np.unique(target)
        labels = labels[labels != 0]
        scores = {}
        for label in labels:
            mask_pred = pred == label
            mask_gt = target == label
            label = str(int(label))
            gt_empty = np.sum(mask_gt) == 0
            pred_empty = np.sum(mask_pred) == 0

            # iou jaccard not defined if both are empty ( 0/0 situation)
            if gt_empty and pred_empty:
                scores[label] = 1.0
            else:
                if pixel_weight is None or pixel_weight[label] is None:
                    intersection = np.logical_and(pred, target)
                    union = np.logical_or(pred, target)
                    iou_score = np.sum(intersection) / np.sum(union)
                    scores[label] = iou_score
                else:
                    ntt = (pred * target * pixel_weight[label]).sum()
                    (nft, ntf) = MetricsSegmentation.get_tf_ft_values_from_bool_array_with_weights(
                        pred, target, pixel_weight[label]
                    )
                    iou_score = ntt / (nft + ntf + ntt)
                    scores[label] = iou_score
        return scores

    @staticmethod
    def overlap(pred: np.ndarray, target: np.ndarray, pixel_weight: Optional[Dict[int, np.ndarray]] = None) -> Dict:
        """
        Calculates overlap score (|X&Y| / min(|X|,|Y|)) based on predicted and target segmentation mask
        Supports multiclass (semantic segmentation)
        :param pred: single sample prediction matrix ( np.ndarray of any shape of type int/bool) per sample
        :param target: target mask ( np.ndarray of any shape of type int/bool) per sample
        :param pixel_weight: Optional dictionary - key = label ,value = weight per pixel . Each element is  float in range [0-1]
        :return: overlap score
        """
        # extract info for the plot
        labels = np.unique(target)
        labels = labels[labels != 0]
        scores = {}
        for label in labels:
            mask_pred = pred == label
            mask_gt = target == label
            label = str(int(label))
            gt_empty = np.sum(mask_gt) == 0
            pred_empty = np.sum(mask_pred) == 0

            # overlap not defined if both are empty ( 0/0 situation)
            if gt_empty and pred_empty:
                scores[label] = 1.0
            elif gt_empty or pred_empty:
                scores[label] = 0.0
            else:
                if pixel_weight is None or pixel_weight[label] is None:
                    intersection = np.logical_and(pred, target)
                    overlap = np.sum(intersection) / min(np.sum(pred), np.sum(target))
                    scores[label] = overlap
                else:
                    intersection = pred * target * pixel_weight[label]
                    overlap = np.sum(intersection) / min(
                        np.sum(pred * pixel_weight[label]), np.sum(target * pixel_weight[label])
                    )
                    scores[label] = overlap
        return scores

    @staticmethod
    def hausdorff_2d_distance(pred: np.ndarray, target: np.ndarray) -> Dict:
        """
        Calculates 2D hausdorff distance based on predicted and target segmentation mask - works for 2D array only!
        Supports multiclass (semantic segmentation)
        :param pred: single sample prediction matrix 2D  array ( np.ndarray  [H, W] of type int/bool) per sample
        :param target: target mask 2D array ( np.ndarray [H, W]  of type int/bool) per sample
        :return: hausdorff distance
        """
        assert len(pred.shape) == 2 or len(target.shape) == 2
        labels = np.unique(target)
        labels = labels[labels != 0]
        scores = {}
        for label in labels:
            mask_pred = pred == label
            mask_gt = target == label
            label = str(int(label))
            mask_pred = mask_pred.astype(int)
            mask_gt = mask_gt.astype(int)
            gt_empty = np.sum(mask_gt) == 0
            pred_empty = np.sum(mask_pred) == 0

            # hausdorff not defined if both are empty ( 0/0 situation)
            if gt_empty and pred_empty:
                scores[label] = 1.0
            else:
                hausdorff1 = directed_hausdorff(mask_pred, mask_gt)[0]
                hausdorff2 = directed_hausdorff(mask_gt, mask_pred)[0]
                hausdorff = max(hausdorff1, hausdorff2)
                scores[label] = hausdorff
        return scores

    @staticmethod
    def pixel_accuracy(
        pred: np.ndarray, target: np.ndarray, pixel_weight: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict:
        """
        Calculates pixel accuracy score (|X&Y| / |Y| ) based on predicted and target segmentation mask
        Supports multiclass (semantic segmentation)
        :param pred: single sample prediction matrix ( np.ndarray of any shape of type int/bool) per sample
        :param target: target mask ( np.ndarray of any shape of type int/bool) per sample
        :param pixel_weight: Optional dictionary - key = label ,value = weight per pixel . Each element is  float in range [0-1]
        :return: pixel accuracy score
        """
        # extract info for the plot
        labels = np.unique(target)
        labels = labels[labels != 0]
        scores = {}
        for label in labels:
            mask_pred = pred == label
            mask_gt = target == label
            label = str(int(label))
            gt_empty = np.sum(mask_gt) == 0
            pred_empty = np.sum(mask_pred) == 0

            # dice not defined if both are empty ( 0/0 situation)
            if gt_empty and pred_empty:
                scores[label] = 1.0
            else:
                if pixel_weight is None or pixel_weight[label] is None:
                    intersection = np.logical_and(pred, target)
                    pixel_accuracy = np.sum(intersection) / np.sum(target)
                    scores[label] = pixel_accuracy
                else:
                    intersection = pred * target * pixel_weight[label]
                    pixel_accuracy = np.sum(intersection) / np.sum(target * pixel_weight[label])
                    scores[label] = pixel_accuracy
        return scores
