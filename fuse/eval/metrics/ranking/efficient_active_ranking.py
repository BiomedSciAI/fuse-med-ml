import numpy as np
from typing import Callable, List, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ItemStats:
    wins: int = 0
    comparisons: int = 0
    lower_bound: float = 0.0
    upper_bound: float = 1.0


class EfficientRanking:
    def __init__(
        self,
        items: List[Any],
        compare_fn: Callable[[Any, Any], bool],
        confidence: float = 0.95,
        min_comparisons: int = 32,
    ):
        self.items = items
        self.compare = compare_fn
        self.confidence = confidence
        self.min_comparisons = min_comparisons
        self.stats = defaultdict(ItemStats)
        self.total_comparisons = 0

    def _update_bounds(self, item: Any) -> None:
        """Update confidence bounds using Hoeffding's inequality"""
        stats = self.stats[item]
        if stats.comparisons == 0:
            return

        p_hat = stats.wins / stats.comparisons
        epsilon = np.sqrt(np.log(2 / self.confidence) / (2 * stats.comparisons))
        stats.lower_bound = max(0.0, p_hat - epsilon)
        stats.upper_bound = min(1.0, p_hat + epsilon)

    def _compare_with_bounds(self, a: Any, b: Any) -> bool:
        """Compare items using confidence bounds to minimize comparisons"""
        stats_a = self.stats[a]
        stats_b = self.stats[b]

        # If bounds don't overlap, we can decide without comparison
        if stats_a.lower_bound > stats_b.upper_bound:
            return True
        if stats_b.lower_bound > stats_a.upper_bound:
            return False

        # If either item has few comparisons, compare directly
        if (
            stats_a.comparisons < self.min_comparisons
            or stats_b.comparisons < self.min_comparisons
        ):
            result = self.compare(a, b)
            self.total_comparisons += 1

            # Update statistics
            if result:
                stats_a.wins += 1
            else:
                stats_b.wins += 1
            stats_a.comparisons += 1
            stats_b.comparisons += 1

            self._update_bounds(a)
            self._update_bounds(b)

            return result

        # If bounds overlap but items have enough comparisons,
        # use current best estimate
        return stats_a.wins / stats_a.comparisons > stats_b.wins / stats_b.comparisons

    def _adaptive_merge(self, items: List[Any]) -> List[Any]:
        """Merge sort with adaptive sampling"""
        if len(items) <= 1:
            return items

        mid = len(items) // 2
        left = self._adaptive_merge(items[:mid])
        right = self._adaptive_merge(items[mid:])

        # Merge with confidence bounds
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if self._compare_with_bounds(left[i], right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def _quicksort_partition(self, items: List[Any], start: int, end: int) -> int:
        """Partition items around pivot using adaptive sampling"""
        if end - start <= 1:
            return start

        pivot = items[end - 1]
        i = start

        for j in range(start, end - 1):
            if self._compare_with_bounds(items[j], pivot):
                items[i], items[j] = items[j], items[i]
                i += 1

        items[i], items[end - 1] = items[end - 1], items[i]
        return i

    def _adaptive_quicksort(self, items: List[Any], start: int, end: int):
        """Quicksort with adaptive sampling"""
        if end - start <= 1:
            return

        pivot = self._quicksort_partition(items, start, end)
        self._adaptive_quicksort(items, start, pivot)
        self._adaptive_quicksort(items, pivot + 1, end)

    def rank(self, method: str = "merge") -> List[Any]:
        """Rank items using specified method"""
        items = self.items.copy()

        if method == "merge":
            return self._adaptive_merge(items)
        elif method == "quick":
            self._adaptive_quicksort(items, 0, len(items))
            return items
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from scipy.stats import spearmanr
    from functools import partial

    def compare_fn(a, b, noise_rate: float = 0.0):
        # Your comparison function
        # return model.predict(a, b)
        if np.random.random() < noise_rate:
            return np.random.random() < 0.5
        return a < b

    num_samples = 10000

    true_scores = np.arange(0, num_samples, 1)

    to_be_ranked = true_scores.copy()
    np.random.shuffle(to_be_ranked)

    ranker = EfficientRanking(
        to_be_ranked, partial(compare_fn, noise_rate=0.1), confidence=0.95
    )
    ranked_items = ranker.rank(method="merge")  # or 'quick'
    print(f"Total comparisons: {ranker.total_comparisons}")
    sr = spearmanr(ranked_items, true_scores)
    print(f"spearman r = {sr.statistic} p = {sr.pvalue}")
