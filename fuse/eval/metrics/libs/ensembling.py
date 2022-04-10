from typing import Dict, Optional, Sequence, Tuple, Union, Hashable
from fuse.eval.metrics.libs.classification import MetricsLibClass
import numpy as np
from sklearn.utils import resample

class Ensembling:
    """
    Methods for ensembling
    """
    
    @staticmethod
    def ensemble(preds: Sequence[np.ndarray]) -> Dict:
        """
        :param preds: sequence of numpy arrays / floats of shape [NUM_CLASSES]
        """
        
        return MetricsLibClass.convert_probabilities_to_class(pred, operation_point)