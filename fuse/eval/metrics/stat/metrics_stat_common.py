from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

from fuse.eval.metrics.metrics_common import MetricWithCollectorBase
from fuse.eval.metrics.regression.metrics import MetricPearsonCorrelation  # noqa: F401


class MetricUniqueValues(MetricWithCollectorBase):
    """
    Collect the all the categorical values and the number of occurrences
    Result format: list of tuples - each tuple include the value and number of occurrences
    """

    def __init__(self, key: str, **kwargs: dict) -> None:
        super().__init__(key=key, **kwargs)

    def eval(
        self, results: dict[str, Any] = None, ids: Sequence[Hashable] | None = None
    ) -> None:
        values = self._extract_arguments(results, ids)["key"]
        counter = Counter(values)

        return list(counter.items())
