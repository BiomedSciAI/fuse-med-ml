import unittest
from typing import Any, Callable, List


def combined_skip(*decorators: List[Callable]) -> Callable:
    def should_skip() -> bool:
        return any(d.condition for d in decorators)

    def decorator(test_item: Any) -> Callable:
        # Combine skip messages
        skip_messages = [d.reason for d in decorators]
        return unittest.skipIf(should_skip(), " OR ".join(skip_messages))(test_item)

    return decorator
