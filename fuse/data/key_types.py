from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Dict

from fuse.data.patterns import Patterns


class DataTypeBasic(Enum):
    UNKNOWN = -1  # TODO: change to Unknown?


class TypeDetectorBase(ABC):
    @abstractmethod
    def get_type(self, sample_dict: Dict, key: str) -> None:
        """
        Returns the type of key
        The most common implementation can be seen in TypeDetectorPatternsBased.
        """
        raise NotImplementedError

    @abstractmethod
    def verify_type(self, sample_dict: Dict, key: str, types: Sequence[Enum]) -> None:
        """
        Raises exception if key is not one of the types found in types
        """
        raise NotImplementedError


class TypeDetectorPatternsBased(TypeDetectorBase):
    def __init__(self, patterns_dict: Dict[str, Enum]):
        """
        Type detection based on the key (NDict "style" - for example 'data.cc.img')
        get_type ignores the sample_dict completely.
        TODO: provide usage example
        """
        self._patterns_dict = patterns_dict
        self._patterns = Patterns(self._patterns_dict, DataTypeBasic.UNKNOWN)

    def get_type(self, sample_dict: Dict, key: str) -> Any:
        return self._patterns.get_value(key)

    def verify_type(self, sample_dict: Dict, key: str, types: Sequence[Enum]) -> None:
        self._patterns.verify_value_in(key, types)
