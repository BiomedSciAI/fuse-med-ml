import unittest
from typing import Any, Callable, List, Tuple


def _id(obj: Any) -> Any:
    return obj


def skipIfMultiple(*skips: List[Tuple[bool, str]]) -> Callable:
    """
    similar to unittest.skipIf but allows to skip depending on multiple conditions

    example usage:

    @skipIfMultiple(
        (banana>12, "banana is bigger than 12!"),
        (apple<10, "apple is less than 10!"),
    )

    """

    for condition, reason in skips:
        if condition:
            return unittest.skip("skipIfMultiple:: " + reason)
    return _id  # passthrough
