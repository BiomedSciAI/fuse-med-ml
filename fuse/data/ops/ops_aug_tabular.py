import random
import numpy as np
from typing import Optional, Any, Sequence

from fuse.data.ops.op_base import OpBase

from fuse.utils import NDict


class OpAugOneHotWithProb(OpBase):
    """
    Apply an augmentation for an one-hot encoding vector with the following modes:

    mode == "default":
        Switch
    """

    def __call__(
        self,
        sample_dict: NDict,
        key: str,
        prob: float,
        idx: Optional[int] = None,
        freeze_indices: Sequence[int] = [],
        mode: str = "default",
    ) -> NDict:
        """
        :param key: key for the one-hot vector
        :param idx: idx to be change to 1
        :param prob: the probability the the functionality will happen
        :param mode: see class desc
        """

        if prob < 0 or prob > 1:
            raise Exception("prob should be between 0 and 1")

        supported_modes = ["default", "ranking"]
        if mode not in supported_modes:
            raise Exception(f"mode ({mode}) should be in supported modes ({supported_modes}).")

        if mode != "default" and idx is not None:
            raise Exception("specify idx only in default mode")

        if mode == "default" and idx is None:
            raise Exception("in 'default' mode, idx must be provided.")

        if random.random() < prob:  # can also use 'pyprob' library
            one_hot = sample_dict[key]

            if mode == "default":
                # TODO add freeze
                one_hot = np.zeros_like(one_hot)
                one_hot[idx] = 1
                sample_dict[key] = one_hot

            if mode == "ranking":
                idx = np.argmax(one_hot)  # Get the current one-hot value

                if idx in freeze_indices:  # do not augment
                    return sample_dict

                # with prob 0.5 idx += 1, else idx -= 1
                idx = (idx+1) if random.random() < 0.5 else (idx-1)

                # make
                idx = max(idx, 0)
                idx = min(idx, len(one_hot) - 1)

        return sample_dict


class OpAugReplaceWithProb(OpBase):
    """
    Replace key's value in sample_dict to any other value with a given probability.

    for example:
        (OpAugReplaceWithProb(), dict(key="data.input.clinical.age", value=42, prob=0.5)),

        will set 'age' value to 42 with 50% probability.
    """

    def __call__(self, sample_dict: NDict, key: str, value: Any, prob: float) -> NDict:
        """
        :param key:
        :param value:
        :param prob:
        """

        if prob < 0 or prob > 1:
            raise Exception(f"prob ({prob}) should be between 0 and 1.")

        if random.random() < prob:
            sample_dict[key] = value

        return sample_dict
