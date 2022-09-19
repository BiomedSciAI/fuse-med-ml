import random
import numpy as np
from typing import Optional, Sequence

from fuse.data.ops.op_base import OpBase

from fuse.utils import NDict


class OpAugOneHot(OpBase):
    """
    Apply an augmentation for an one-hot encoding vector.
    Op supports the following modes:

    mode == "default":
        replace the current one-hot encoding idx with a given one.

        example of use with OpRandApply as a meta-op:
                (
                    OpRandApply(OpAugOneHot(), probability=0.05),
                    dict(key="data.input.clinical.encoding.sex", idx=2, mode="default"),
                ),
            will change the one-hot index to 2 with probability of 0.05 (Here index 2 is N/A).

    mode == "ranking":
        replace the current one-hot encoding vector with a new one where the '1's idx will be +=1 the original one.
        meaning, (0,0,1,0) will turn into (0,1,0,0) or (0,0,0,1) (with prob 0.5).

        example of use with OpRandApply as a meta-op:
                (
                    OpRandApply(OpAugOneHot(), probability=0.05),
                    dict(key="data.input.clinical.encoding.age", mode="ranking", freeze_indices=[6]),
                ),
            will change the index of the age encoding += 1 with a probability of 0.05, but do nothing if the one-hot encoding has a '1' in the 6th index.
    """

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        idx: Optional[int] = None,
        freeze_indices: Sequence[int] = [],
        mode: str = "default",
    ) -> NDict:
        """
        :param key: key for a one-hot numpy array vector.
        :param idx: in "default" mode: idx to be change to 1
        :param freeze_indices: in "ranking" mode: sequence of indices such that if one-hot vector has a '1' in one of those indices, the augmentation won't be executed.
        :param mode: see class desc
        """

        supported_modes = ["default", "ranking"]
        if mode not in supported_modes:
            raise Exception(f"mode ({mode}) should be in supported modes ({supported_modes}).")

        if mode != "default" and idx is not None:
            raise Exception("specify idx only in default mode")

        if mode == "default" and idx is None:
            raise Exception("in 'default' mode, idx must be provided.")

        one_hot = sample_dict[key]
        res_one_hot = np.zeros_like(one_hot)

        if mode == "ranking":
            idx = np.argmax(one_hot)  # Get the current one-hot value

            if idx in freeze_indices:  # do not augment
                return sample_dict

            # idx +- 1 with probability of 0.5
            idx = (idx + 1) if random.random() < 0.5 else (idx - 1)

        # make sure idx in range
        idx = max(idx, 0)
        idx = min(idx, len(one_hot) - 1)

        # set one-hot
        res_one_hot[idx] = 1
        sample_dict[key] = res_one_hot

        return sample_dict
