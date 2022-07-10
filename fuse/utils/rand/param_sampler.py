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
from abc import ABC, abstractmethod

from typing import Any, Optional, Sequence, List, Tuple
import random

import numpy as np


class ParamSamplerBase(ABC):
    """
    Base class for param sampler
    """

    @abstractmethod
    def sample(self) -> Any:
        """
        :return: the sampled value
        """
        raise NotImplementedError


class Uniform(ParamSamplerBase):
    def __init__(self, min: float, max: float):
        """
        Uniform distribution between min and max
        """
        super().__init__()

        # store input parameters
        self.min = min
        self.max = max

    def sample(self) -> float:
        """
        :return: random float in range [min-max]
        """
        return random.uniform(self.min, self.max)

    def __str__(self):
        return f"Uniform [{self.min} - {self.max}] "


class RandInt(ParamSamplerBase):
    def __init__(self, min: int, max: int):
        """
        Uniform integer distribution between min and max
        """
        super().__init__()

        # store input parameters
        self.min = min
        self.max = max

    def sample(self) -> int:
        """
        :return: random int in range [min-max]
        """
        return random.randint(self.min, self.max)

    def __str__(self):
        return f"RandInt [{self.min} - {self.max}] "


class RandBool(ParamSamplerBase):
    def __init__(self, probability: float):
        """
        Random boolean according to the given probability
        """
        super().__init__()

        # store input parameters
        self.probability = probability

    def sample(self) -> bool:
        """
        :return: random boolean according to probability
        """
        return random.uniform(0, 1) <= self.probability

    def __str__(self):
        return f"RandBool p={self.probability}] "


class Choice(ParamSamplerBase):
    def __init__(self, seq: Sequence, probabilities: Optional[List[float]] = None, k: int = 0):
        """
        Random choice out of a sequence
        Return a k sized list of population elements chosen with replacement
        If k=0 return a single element instead of a list
        """
        super().__init__()

        # store input parameters
        self.seq = seq
        self.probabilities = probabilities
        self.k = k

    def sample(self) -> Any:
        """
        :return: random element according to probabilities
        """
        if self.k == 0:
            return random.choices(self.seq, weights=self.probabilities)[0]
        else:
            return random.choices(self.seq, weights=self.probabilities, k=self.k)

    def __str__(self):
        return f"Choice seq={self.seq}, w={self.probabilities}] "


class Gaussian(ParamSamplerBase):
    """
    Gaussian noise
    """

    def __init__(self, shape: Tuple[int, ...], mean: float, std: float):
        """
        :param shape: patch size of the required noise
        :param mean: mean of the gauss noise
        :param std: std of the gauss noise
        """
        super().__init__()

        # store input parameters
        self.shape = shape
        self.mean = mean
        self.std = std

    def sample(self) -> np.ndarray:
        """
        :return: random gaussian noise
        """
        return self.std * np.random.randn(*list(self.shape)) + self.mean


def draw_samples_recursively(data: Any) -> Any:
    """
    Generate a copy of the data structure, replacing each ParamSamplerBase with a random sample.

    :param data:data_structure: recursively looking for ParamSamplerBase in a dictionary and a sequence
    :return: See above
    """
    # if a dictionary return a copy of the dictionary try to sample each value recursively
    if isinstance(data, dict):
        data_dict: dict = data.copy()
        for key in data_dict:
            data_dict[key] = draw_samples_recursively(data_dict[key])
        return data_dict

    # if a list  return a copy of the list try to sample each element recursively
    if isinstance(data, list):
        data_lst: list = data[:]
        for ii in range(len(data_lst)):
            data_lst[ii] = draw_samples_recursively(data_lst[ii])
        return data_lst

    # if a tuple  return a copy of the tuple try to sample each element recursively
    if isinstance(data, Tuple):
        data_tuple = tuple((draw_samples_recursively(data[ii]) for ii in range(len(data))))
        return data_tuple

    # if ParamSamplerBase, sample a number
    if isinstance(data, ParamSamplerBase):
        data_sampler: ParamSamplerBase = data
        return data_sampler.sample()

    # otherwise return the original data
    return data
