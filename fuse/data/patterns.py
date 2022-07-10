from collections import OrderedDict
from typing import Any, Sequence
import re


class Patterns:
    """
    Utility to match a string to a pattern.
    Typically used to infer data type from key in sample_dict
    """

    def __init__(self, patterns_dict: OrderedDict, default_value: Any = None):
        """
        :param patterns_dict: ordered dictionary, the key is a regex expression.
                        The value of the first matched key will be returned.
        Example:
        patterns = {
            r".*img$": DataType.IMAGE,
            r".*seg$": DataType.SEG,
            r".*bbox$": DataType.BBOX,
            r".*$": DataType.UNKNOWN
        }
        pp = Patterns(patterns)
        print(pp.get_type("data.cc.img")) -> DataType.IMAGE
        print(pp.get_type("data.cc_img")) -> DataType.IMAGE
        print(pp.get_type("data.img_seg")) -> DataType.SEG
        print(pp.get_type("data.imgseg")) -> DataType.SEG
        print(pp.get_type("data")) -> DataType.UNKNOWN
        print(pp.get_type("bbox")) -> DataType.BBox
        print(pp.get_type("a.bbox")) -> DataType.BBOX

        :param default_value: value to return in case there is not match
        """
        self._patterns = patterns_dict
        self._default_value = default_value

    def get_value(self, key: str) -> Any:
        """
        :param key: string to match
        :return: the first value from patterns with pattern that match to key
        """
        for pattern in self._patterns:
            if re.match(pattern, key) is not None:
                return self._patterns[pattern]

        return self._default_value

    def verify_value_in(self, key: str, values: Sequence[Any]) -> None:
        """
        Raise an exception of the matched value not in values
        :param key: string to match
        :param values: list of supported values
        :return: None
        """
        val_type = self.get_value(key)
        if val_type not in values:
            raise ValueError(
                f"key {key} mapped to unsupported type {val_type}.\n List of supported types {values} \n Patterns {self._patterns}"
            )
