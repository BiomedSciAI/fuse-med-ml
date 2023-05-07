from typing import Dict, Sequence, Union
from fuse.eval.metrics.libs.classification import MetricsLibClass
import numpy as np


class Thresholding:
    """
    Methods for thresholding
    """

    @staticmethod
    def apply_thresholds(
        pred: Sequence[np.ndarray], operation_point: Union[float, Sequence[float], None] = None
    ) -> Dict:
        """
        :param pred: sequence of numpy arrays / floats of shape [NUM_CLASSES]
        :param operation_point: Optional. If specified will be used to convert probabilities to class prediction.
                                          Options:
                                          * None - argmax(probability) - default mode
                                          * float - used as a threshold for the positive class in binary classification
                                          * [(<class_index_first>, <threshold_first>), ...,(<class_index_last>, <threshold_last>)] -
                                            the class prediction will be the first class in which the probability cross the threshold.
                                            If no class probability crossed the threshold the predicated class will be set to -1
        """

        return MetricsLibClass.convert_probabilities_to_class(pred, operation_point)
