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
        compare_batch_fn: Callable[[List[Any], List[Any]], List[bool]],
        confidence: float = 0.95,
        min_comparisons: int = 32,
        batch_size: int = 32,
    ):
        self.items = items
        self.compare_batch = compare_batch_fn
        self.confidence = confidence
        self.min_comparisons = min_comparisons
        self.batch_size = batch_size
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

    def _compare_batch_with_bounds(
        self, batch_a: List[Any], batch_b: List[Any]
    ) -> List[bool]:
        """Compare batches of items using confidence bounds to minimize comparisons"""
        # Prepare results and track which comparisons need direct evaluation
        results = [False] * len(batch_a)
        direct_compare_indices = []
        direct_compare_a = []
        direct_compare_b = []

        for i, (a, b) in enumerate(zip(batch_a, batch_b)):
            stats_a = self.stats[a]
            stats_b = self.stats[b]

            # If bounds don't overlap, we can decide without comparison
            if stats_a.lower_bound > stats_b.upper_bound:
                results[i] = True
            elif stats_b.lower_bound > stats_a.upper_bound:
                results[i] = False
            elif (
                stats_a.comparisons < self.min_comparisons
                or stats_b.comparisons < self.min_comparisons
            ):
                # Need direct comparison
                direct_compare_indices.append(i)
                direct_compare_a.append(a)
                direct_compare_b.append(b)
            else:
                # Use current win rates if bounds overlap
                results[i] = (
                    stats_a.wins / stats_a.comparisons
                    > stats_b.wins / stats_b.comparisons
                )

        # Perform batch comparisons for items needing direct evaluation
        if direct_compare_a:
            batch_results = self.compare_batch(direct_compare_a, direct_compare_b)
            self.total_comparisons += len(batch_results)

            # Update results and statistics
            for idx, (a, b, result) in enumerate(
                zip(direct_compare_a, direct_compare_b, batch_results)
            ):
                orig_idx = direct_compare_indices[idx]
                results[orig_idx] = result

                stats_a = self.stats[a]
                stats_b = self.stats[b]

                if result:
                    stats_a.wins += 1
                else:
                    stats_b.wins += 1
                stats_a.comparisons += 1
                stats_b.comparisons += 1

                self._update_bounds(a)
                self._update_bounds(b)

        return results

    def _adaptive_merge(self, items: List[Any]) -> List[Any]:
        """Merge sort with adaptive batch sampling"""
        if len(items) <= 1:
            return items

        mid = len(items) // 2
        left = self._adaptive_merge(items[:mid])
        right = self._adaptive_merge(items[mid:])

        # Merge with confidence bounds
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            # Process batches to reduce number of comparisons
            batch_size = min(self.batch_size, len(left) - i, len(right) - j)

            batch_left = left[i : i + batch_size]
            batch_right = right[j : j + batch_size]

            # Perform batch comparisons
            batch_results = self._compare_batch_with_bounds(batch_left, batch_right)

            # Merge based on batch results
            for k, is_left_winner in enumerate(batch_results):
                if is_left_winner:
                    result.append(batch_left[k])
                    i += 1
                else:
                    result.append(batch_right[k])
                    j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def rank(self, method: str = "merge") -> List[Any]:
        """Rank items using specified method"""
        items = self.items.copy()

        if method == "merge":
            return self._adaptive_merge(items)
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from scipy.stats import spearmanr

    # from functools import partial

    def compare_fn(a:Any, b:Any, noise_rate: float = 0.0) -> bool:
        # Your comparison function
        # return model.predict(a, b)
        if np.random.random() < noise_rate:
            return np.random.random() < 0.5
        return a < b

    def batch_compare(batch_a, batch_b) -> List:
        # Return a list of boolean results for batch comparisons
        return [compare_fn(a, b) for a, b in zip(batch_a, batch_b)]

    num_samples = 10000

    true_scores = np.arange(0, num_samples, 1)

    to_be_ranked = true_scores.copy()
    np.random.shuffle(to_be_ranked)

    ranker = EfficientRanking(to_be_ranked, batch_compare)
    ranked_items = ranker.rank()

    print(f"Total comparisons: {ranker.total_comparisons}")
    sr = spearmanr(ranked_items, true_scores)
    print(f"spearman r = {sr.statistic} p = {sr.pvalue}")
