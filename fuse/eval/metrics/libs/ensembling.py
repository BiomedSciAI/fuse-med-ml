from typing import Dict, Optional, Sequence
import numpy as np
import scipy


class Ensembling:
    """
    Methods for ensembling
    """

    @staticmethod
    def ensemble(preds: Sequence[np.ndarray], method: Optional[str] = "average") -> Dict:
        """
        :param preds: sequence of numpy arrays / floats of shape [NUM_CLASSES]
        :params method: Ensembling method. 'average', or 'voting'
            if 'average', the predictions are assumed to be continuous (probabilities or regression output)
            if 'voting', the predictions are assumed to be class predictions (integers)
        """
        if isinstance(preds, Sequence):
            preds = np.stack(preds)

        # ensemble
        if method.lower() in ("average", "mean"):
            preds_ensembled = np.mean(preds, 1)
        elif method.lower() in ("vote", "voting"):
            assert len(preds.shape) == 2 or (len(preds.shape) == 3 and preds.shape[2] == 1)
            if len(preds.shape) == 3:
                preds = np.squeeze(preds)
            preds_ensembled = scipy.stats.mode(preds, axis=1)[0]
        else:
            raise ValueError("'average' or 'voting' methods are supported for ensembling")

        return preds_ensembled
