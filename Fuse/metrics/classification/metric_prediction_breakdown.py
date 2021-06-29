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

from typing import Dict, List, Tuple, Callable
import numpy as np
import pandas as pd

from Fuse.metrics.metric_base import FuseMetricBase
from Fuse.metrics.metrics_toolbox import FuseMetricsToolBox


class FuseMetricPredictionBreakdown(FuseMetricBase):
    """
    Multi class version for to breakdown prediction to subgroups/subtypes.

    """

    def __init__(self, pred_name: str, target_name: str, breakdown_class_name: str,
                 class_names: List = None, class_thresholds: List[Tuple] = None, **kwargs):
        """
        :param pred_name:       batch_dict key for predicted output (e.g., class probabilities after softmax)
        :param target_name:     batch_dict key for target (e.g., ground truth label)
        :param class_names:     Optional - name for each class otherwise the class index will be used.
        :param breakdown_class_name: sybtype of each sample,the breakdown will be according to this lable.
        :param class_thresholds: Optional - threshold for each class, the order of the list is the order in which class
                            is assigned, Each element in the list is a tuple of (class_idx, class_threshold)
                            if None: the class is set by argmax
        """
        super().__init__(pred_name, target_name, breakdown_class_name=breakdown_class_name, **kwargs)
        self._class_names = class_names
        self._class_thresholds = class_thresholds

    def process(self) -> Dict[str, float]:
        """
        Returns a string table with class prediction breakdown according to sample subtype
        """

        epoch_preds = np.array(self.epoch_preds)
        epoch_targets = np.array(self.epoch_targets)
        epoch_breakdown = np.array(self.collected_data['breakdown_class'])

        # first find operation points
        if isinstance(self._class_thresholds, Callable):
            class_thresholds = self._class_thresholds(epoch_preds, epoch_targets)
        else:
            class_thresholds = self._class_thresholds

        # get class predictions
        epoch_class_preds = FuseMetricsToolBox.convert_probabilities_to_class(epoch_preds, thresholds=class_thresholds)

        # get class names
        num_classes = epoch_preds[0].shape[0]
        class_names = FuseMetricsToolBox.get_class_names(num_classes, self._class_names)

        # get breakdown classes (sub types)
        breakdown_classes = sorted(list(set(epoch_breakdown)), key=lambda x: str(x))
        breakdown_classes_count = self.get_list_type_count(breakdown_classes, epoch_breakdown)

        # fill in a 2D table of breakdown_cls x predicted cls_
        breakdown_count_per_class = {str(cls): [] for cls in breakdown_classes}
        for breakdown_cls in breakdown_classes:
            breakdown_cls_name = str(breakdown_cls)
            for class_idx, class_name in enumerate(class_names):
                detect_count = np.logical_and(np.where(epoch_breakdown == breakdown_cls, 1, 0),
                                              np.where(epoch_class_preds == class_idx, 1, 0)).sum()
                all_count = breakdown_classes_count[breakdown_cls]
                count_percent = '(' + str(round(detect_count / int(all_count) * 100, 1)) + '%)'
                breakdown_count_per_class[breakdown_cls_name].append(str(detect_count) + '/' + all_count + count_percent)

        return pd.DataFrame(breakdown_count_per_class, index=class_names).transpose().to_string()

    @staticmethod
    def get_list_type_count(breakdown_classes, epoch_breakdown) -> Dict:
        return {breakdown_cls: str(list(epoch_breakdown).count(breakdown_cls)) for breakdown_cls in breakdown_classes}
