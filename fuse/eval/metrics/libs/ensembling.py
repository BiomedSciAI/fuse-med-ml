from typing import Dict, Optional, Sequence, Tuple, Union, Hashable
from fuse.eval.metrics.libs.classification import MetricsLibClass
import numpy as np
from sklearn.utils import resample
import pandas as pd


class Ensembling:
    """
    Methods for ensembling
    """

    @staticmethod
    def ensemble(preds: Sequence[np.ndarray], method: Optional[str] = 'average') -> Dict:
        """
        :param preds: sequence of numpy arrays / floats of shape [NUM_CLASSES]
        :params method: Ensembling method. 'average', 'median' or 'voting'
        """
        if isinstance(preds, Sequence):
            preds = np.stack(preds)

        # ensemble
        if method.lower() in ('average', 'mean'):
            preds_ensembled = np.mean(preds, 1)  
        else:
            raise ValueError("Currently only 'average' method is supported for ensembling")

        return preds_ensembled
