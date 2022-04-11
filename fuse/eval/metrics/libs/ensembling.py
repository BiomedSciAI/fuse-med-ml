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
        
        softmax = np.vstack(softmax)
        softmax = np.mean(softmax, 0) # ensemble
        return softmax