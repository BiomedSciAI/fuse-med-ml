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

import secrets
from abc import ABC, abstractmethod
from typing import Any, Sequence, List, Tuple

import torch
from torch import Tensor

secretsGenerator = secrets.SystemRandom()


class FuseUtilsParamSamplerBase(ABC):
    """
    Base class for param sampler
    """

    @abstractmethod
    def sample(self) -> Any:
        """
        :return: the sampled value
        """
        raise NotImplementedError


class FuseUtilsParamSamplerUniform(FuseUtilsParamSamplerBase):
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
        return secretsGenerator.uniform(self.min, self.max)

    def __str__(self):
        return f'RandUniform [{self.min} - {self.max}] '


class FuseUtilsParamSamplerRandInt(FuseUtilsParamSamplerBase):
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
        return secretsGenerator.randint(self.min, self.max)

    def __str__(self):
        return f'RandInt [{self.min} - {self.max}] '


class FuseUtilsParamSamplerRandBool(FuseUtilsParamSamplerBase):
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
        return secretsGenerator.uniform(0, 1) <= self.probability

    def __str__(self):
        return f'RandBool p={self.probability}] '


class FuseUtilsParamSamplerChoice(FuseUtilsParamSamplerBase):

    def __init__(self, seq: Sequence, probabilities: List[float], k: int = 0):
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
            return secretsGenerator.choices(self.seq, weights=self.probabilities)[0]
        else:
            return secretsGenerator.choices(self.seq, weights=self.probabilities, k=self.k)

    def __str__(self):
        return f'RandChoice seq={self.seq}, w={self.probabilities}] '


def sample_all(data: Any) -> Any:
    """
    Generate a copy of the data structure, replacing each FuseUtilsParamSamplerBase with a random sample.

    :param data:data_structure: recursivelly looking for FuseUtilsParamSamplerBase in a dictionary and a sequence
    :return: See above
    """
    # if a dictionary return a copy of the dictionary try to sample each value recursively
    if isinstance(data, dict):
        data_dict: dict = data.copy()
        for key in data_dict:
            data_dict[key] = sample_all(data_dict[key])
        return data_dict

    # if a list  return a copy of the list try to sample each element recursively
    if isinstance(data, list):
        data_lst: list = data[:]
        for ii in range(len(data_lst)):
            data_lst[ii] = sample_all(data_lst[ii])
        return data_lst

    # if a tuple  return a copy of the tuple try to sample each element recursively
    if isinstance(data, Tuple):
        data_tuple = tuple((sample_all(data[ii]) for ii in range(len(data))))
        return data_tuple

    # if FuseUtilsParamSamplerBase, sample a number
    if isinstance(data, FuseUtilsParamSamplerBase):
        data_sampler: FuseUtilsParamSamplerBase = data
        return data_sampler.sample()

    # otherwise return the original data
    return data


class FuseUtilsParamSamplerGaussianPatch(FuseUtilsParamSamplerBase):
    def __init__(self, shape: Tuple[int, ...], mean: float, std: float):
        """
        Gaussian noise
        """
        super().__init__()

        # store input parameters
        self.shape = shape
        self.mean = mean
        self.std = std

    def sample(self) -> Tensor:
        """
        :return: random gaussian noise
        """
        return self.std * torch.randn(self.shape) + self.mean
