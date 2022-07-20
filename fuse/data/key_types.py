from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Sequence
from fuse.data.patterns import Patterns


class DataTypeBasic(Enum):
    UNKNOWN = -1  # TODO: change to Unknown?


class TypeDetectorBase(ABC):
    @abstractmethod
    def get_type(self, sample_dict: Dict, key: str):
        """
        Returns the type of key
        The most common implementation can be seen in TypeDetectorPatternsBased.
        """
        raise NotImplementedError

    @abstractmethod
    def verify_type(self, sample_dict: Dict, key: str, types: Sequence[Enum]):
        """
        Raises exception if key is not one of the types found in types
        """
        raise NotImplementedError


class TypeDetectorPatternsBased(TypeDetectorBase):
    def __init__(self, patterns_dict: Dict[str, Enum]):
        """
        type detection based on the key (NDict "style" - for example 'data.cc.img')
        get_type ignores the sample_dict completely.
        TODO: provide usage example
        """
        self._patterns_dict = patterns_dict
        self._patterns = Patterns(self._patterns_dict, DataTypeBasic.UNKNOWN)

    def get_type(self, sample_dict: Dict, key: str):
        return self._patterns.get_value(key)

    def verify_type(self, sample_dict: Dict, key: str, types: Sequence[Enum]):
        self._patterns.verify_value_in(key, types)
