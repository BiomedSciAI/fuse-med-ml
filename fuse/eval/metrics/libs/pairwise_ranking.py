from functools import partial
from typing import Any, Callable, List

import numpy as np
from choix import ilsr_pairwise_dense


class PairwiseRanking:
    """
    Rank a set of items [a,b,c,...n] based on pairwise comparisons.
    The pairwise comparisons should be given as a matrix, and may be complete (n^2) or parital.

    The following ranking methods are implemented:
    - BTM : Bradely Terry Model, see https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model, using choix implementations
    see https://choix.lum.li/en/latest/api.html#choix.ilsr_pairwise_dense.
    """

    def __init__(self, items: List[Any], comparison_matrix: np.array):
        """
        items: items to be ranked
        comparison_matrix (np.array =2D): a square matrix describing the pairwise-comparison results, comparison_matrix[i,j] contains the number
        of times that item i wins against item j (or just "1"). "0" means missing info or "never".
        This can be used for comparing many rounds of i vs j, as well as single comparisons per pair, or for some of the pairs.
        """

        self.items = items
        self.comparison_matrix = comparison_matrix

    def rank(self, method: str = "BTM") -> List[Any]:
        """Rank items using specified method
        BTM: Bradley Terry model, using maximum liklihood inference I-LSR algortihm
        see: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
        and https://choix.lum.li/en/latest/api.html#choix.ilsr_pairwise_dense
        """

        if method == "BTM":
            params = ilsr_pairwise_dense(self.comparison_matrix, alpha=0.001)
            ranked_items = [item for _, item in sorted(zip(params, self.items))]
            return ranked_items
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":

    import time

    from scipy.stats import spearmanr

    def compare_fn(a: Any, b: Any, noise_rate: float = 0.0) -> bool:
        """Pairwise comparison with option to simulate noise and missing values"""
        if np.random.random() < noise_rate:
            return np.random.random() < 0.5
        return a < b

    def compare_all_pairs(
        items: List[Any],
        compare_fn: Callable[[Any, Any], bool],
        missing_rate: float = 0,
    ) -> list:
        """
        compare all pairs, set the comparison matrix to be 1 if i<j and 0 otherwise
        """
        data = np.zeros((len(items), len(items)))
        for i in range(len(items)):
            for j in range(len(items)):
                if np.random.random() > missing_rate:
                    if compare_fn(items[i], items[j]):
                        data[j, i] = 1  # items[j] wins
        return data

    num_samples = 200

    true_scores = np.arange(0, num_samples, 1)

    to_be_ranked = true_scores.copy()
    np.random.shuffle(to_be_ranked)

    compare_mat = compare_all_pairs(
        to_be_ranked, partial(compare_fn, noise_rate=0.5), missing_rate=0.6
    )

    ranker = PairwiseRanking(to_be_ranked, compare_mat)
    start_time = time.time()

    ranked_items = ranker.rank(method="BTM")
    end_time = time.time()

    print(f"Elapsed time: {(end_time-start_time):.4f} seconds")

    sr = spearmanr(ranked_items, true_scores)
    print(f"spearman r = {sr.statistic} p = {sr.pvalue}")
