from functools import partial
from typing import Any, Callable, List

import numpy as np
from choix import ilsr_pairwise


class CompletePairwiseRanking:
    """
    Rank a set of items [a,b,c,...n] based on a pairwise compartor able to compare items i and j.
    It first uses the pairwise ranked to rank all pairs (n^2 calls), and then uses the pairwise ranking to rank the whole list.
    Caution: Running time is n^2, this method will be slow on long inputs or slow pairwise comparators.

    The following ranking methods are implemented:
    - BTM : Bradely Terry Model, see https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    """

    def __init__(
        self,
        items: List[Any],
        compare_pairwise_fn: Callable[[Any, Any], bool],
    ):
        """
        items: items to be ranked
        compare_pairwise_fn: a function like this:

            def pairwise_compare(a:int, b:int) -> bool:
                # return boolean - should have a lower ranking number than b?
                # an AI model (or anything) that predicts it.
                # note: for binding affinity the convention is that being ranked lower means stronger binding.
                # for example, rank 0 is the strongest binder in the list.
        """

        self.items = items
        self.compare = compare_pairwise_fn

    def rank(self, method: str = "BTM") -> List[Any]:
        """Rank items using specified method
        BTM: Bradley Terry model, using maximum liklihood inference I-LSR algortihm
        see: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
        and https://choix.lum.li/en/latest/api.html#choix.ilsr_pairwise
        """

        data = self._compare_all_pairs()

        if method == "BTM":
            params = ilsr_pairwise(len(self.items), data, alpha=0.001)
            ranked_items = [item for _, item in sorted(zip(params, self.items))]
            return ranked_items
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compare_all_pairs(self) -> list:
        data = []
        for i, j in [
            (i, j)
            for i in range(len(self.items))
            for j in range(len(self.items))
            if i != j
        ]:
            if self.compare(self.items[i], self.items[j]):
                data.append((j, i))  # items[j] wins
            else:
                data.append((i, j))  # items[i] wins
        return data


if __name__ == "__main__":

    from scipy.stats import spearmanr

    def compare_fn(a: Any, b: Any, noise_rate: float = 0.0) -> bool:
        # Your comparison function
        # return model.predict(a, b)
        if np.random.random() < noise_rate:
            return np.random.random() < 0.5
        return a < b

    num_samples = 100

    true_scores = np.arange(0, num_samples, 1)

    to_be_ranked = true_scores.copy()
    np.random.shuffle(to_be_ranked)

    ranker = CompletePairwiseRanking(to_be_ranked, partial(compare_fn, noise_rate=0.5))
    ranked_items = ranker.rank(method="BTM")
    sr = spearmanr(ranked_items, true_scores)
    print(f"spearman r = {sr.statistic} p = {sr.pvalue}")
