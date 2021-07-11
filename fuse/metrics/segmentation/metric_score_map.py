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

from typing import Optional, Union, Dict, Sequence

import numpy as np

from fuse.metrics.metric_base import FuseMetricBase
from Fuse.utils.utils_image_processing import FuseUtilsImageProcessing


class FuseMetricScoreMap(FuseMetricBase):
    """
    Segmentation metrics.
    """

    def __init__(self,
                 pred_name: str,
                 target_name: str,
                 hard_threshold: bool = False,
                 threshold: float = 0.0,
                 class_names: Optional[Sequence[str]] = None):
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax).
                                Expected Tensor shape = [batch, num_classes, height, width]
        :param target_name:     batch_dict key for target (e.g., ground truth label). Expected Tensor shape = [batch, height, width]
        :param hard_threshold:  Boolean for using predictions as probabilities for dice calculations of turning predicitons to binary output.
        :param threshold:       Threshold for creating a boolean image from probabilities.

        :param class_names:     Optional - class names
        """
        super().__init__(pred_name, target_name)
        self._class_names = class_names
        self.hard_threshold = hard_threshold
        self.threshold = threshold

    def process(self) -> Union[float, Dict[str, float]]:
        num_classes = self.collected_data['pred'][0].shape[0]

        raw_metrics_per_class = []  # tp/fp/fn accumulators per class
        for cls_idx in range(num_classes + 1):  # Adding one extra class for average
            raw_metrics_per_class.append({'tp': 0, 'fp': 0, 'fn': 0})

        for sample_idx in range(len(self.collected_data['pred'])):
            targets = self.collected_data['target'][sample_idx]  # shape = [height, width]
            preds = self.collected_data['pred'][sample_idx]  # shape = [num_classes, height, width]

            targets_height, targets_width = targets.shape
            num_classes, preds_height, preds_width = preds.shape

            # Resize targets (if needed) to match score map
            # ==============================================
            if (targets_height > preds_height) or (targets_width > preds_width):
                targets = FuseUtilsImageProcessing.preserve_range_resize(targets, target_shape=(preds_height, preds_width))

            # Calculate metrics per class (start at index 0 (background) to cover cases where no mask exists)
            # ================================================================
            for cls_idx in range(0, num_classes):
                cls_target = (targets == cls_idx)
                cls_pred = preds[cls_idx]

                p = np.sum(cls_target)
                n = np.sum(~cls_target)
                if self.hard_threshold:
                    cls_pred_thresh = cls_pred > self.threshold
                    tp = np.sum(cls_pred_thresh[cls_target])  # sum intersection pixels - true positives
                    fp = np.sum(cls_pred_thresh[~cls_target])
                else:
                    tp = np.sum(cls_pred[cls_target])  # sum intersection pixels - true positives
                    fp = np.sum(cls_pred[~cls_target])  # sum pixels outside ground truth area - false positives
                tn = n - fp  # not used...
                fn = p - tp

                raw_metrics_per_class[cls_idx]['tp'] += tp
                raw_metrics_per_class[cls_idx]['fp'] += fp
                raw_metrics_per_class[cls_idx]['fn'] += fn

                # Collect global sums for the extra 'average' class
                raw_metrics_per_class[num_classes]['tp'] += tp
                raw_metrics_per_class[num_classes]['fp'] += fp
                raw_metrics_per_class[num_classes]['fn'] += fn

        if self._class_names is None:
            self._class_names = ['class_' + str(i) for i in range(1, num_classes)]
        self._class_names.append('micro_avg')

        result = {}
        for cls_idx in range(1, num_classes + 1):
            tp = raw_metrics_per_class[cls_idx]['tp']
            fn = raw_metrics_per_class[cls_idx]['fn']
            fp = raw_metrics_per_class[cls_idx]['fp']

            if (fn + tp + fp) > 0:
                iou = tp / (fn + tp + fp)
                dice = (2 * tp) / (fn + 2 * tp + fp)
            else:
                iou = np.nan
                dice = np.nan

            class_name = self._class_names[cls_idx - 1]
            result[class_name + '_iou'] = iou
            result[class_name + '_dice'] = dice

        return result
