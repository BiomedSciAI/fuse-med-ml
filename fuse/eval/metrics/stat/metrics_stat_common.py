from typing import Any, Dict, Hashable, Optional, Sequence
from collections import Counter
from fuse.eval.metrics.metrics_common import MetricWithCollectorBase


class MetricUniqueValues(MetricWithCollectorBase):
    """
    Collect the all the categorical values and the number of occurrences
    Result format: list of tuples - each tuple include the value and number of occurrences
    """

    def __init__(self, key: str, **kwargs) -> None:
        super().__init__(key=key, **kwargs)

    def eval(self, results: Dict[str, Any] = None, ids: Optional[Sequence[Hashable]] = None) -> None:
        values = self._extract_arguments(results, ids)["key"]
        counter = Counter(values)

        return list(counter.items())
