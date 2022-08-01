from typing import Sequence, Optional, Hashable
import numpy as np
import copy


class PerSampleData:
    """
    Basic class for sampling data at given sample ids.
    Used in order to maintain sample ids alignment with the source data when
    applying metrics and operations which operate per sample and combining them with
    other metrics in a pipeline.
    """

    def __init__(self, data: Sequence[np.ndarray], ids: Sequence[Hashable]):
        self._data = data
        self._ids = ids

    def __call__(self, ids: Optional[Sequence[Hashable]] = None) -> Sequence[np.ndarray]:
        if ids is None:
            return copy.copy(self._data)
        else:
            # convert required ids to permutation
            original_ids = self._ids
            required_ids = ids
            permutation = [original_ids.index(sample_id) for sample_id in required_ids]

            return [self._data[i] for i in permutation]
