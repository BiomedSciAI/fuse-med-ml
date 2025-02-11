"""
(C) Copyright 2023 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Jan 11, 2023

"""

import random
import unittest
from functools import partial
from typing import Any, Callable, List

import numpy as np
from scipy.stats import spearmanr

from fuse.eval.metrics.libs.efficient_active_ranking_pairwise_model import (
    EfficientRanking,
)
from fuse.eval.metrics.libs.efficient_ranking_batch_rank import aggregate_rankings
from fuse.eval.metrics.libs.pairwise_ranking import PairwiseRanking


def pairwise_compare_fn(a: Any, b: Any, noise_rate: float = 0.0) -> bool:
    # Your comparison function
    # return model.predict(a, b)
    if np.random.random() < noise_rate:
        return np.random.random() < 0.5
    return a < b


def compare_all_pairs(
    items: List[Any], compare_fn: Callable[[Any, Any], bool], missing_rate: float = 0
) -> list:
    """
    compare all pairs, set the comparison matrix to be 1 if i<j.
    missing_values_rate(double) - rate of missing pairs, stored as 0 in the matrix
    """
    data = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(len(items)):
            if np.random.random() > missing_rate:
                if compare_fn(items[i], items[j]):
                    data[j, i] = 1  # items[j] wins

    return data


def list_rank_func(items: List, number_of_random_flipped: int = 0) -> List:
    ans = sorted(items)

    if number_of_random_flipped > 0:
        length = len(items)
        for _ in range(number_of_random_flipped):
            i = np.random.randint(0, length)
            val_i = items[i]
            j = np.random.randint(0, length)
            val_j = items[j]
            items[i] = val_j
            items[j] = val_i

    return ans


# an example that uses a pairwise compare function


def convert_pairwise_to_ranker(
    pairwise_model: Callable[[Any, Any], bool]
) -> Callable[[List], List]:
    """
    A helper function that converts a pairwise model to a subsample ranker of length 2,
    to support using pairwise model in `aggregate_rankings()`

    pairwise_model: a function that returns true if the item provided as first argument should be ranked higher than the item provided as the second argument
    """

    def build_func(items: List) -> List:
        assert 2 == len(items)
        if pairwise_model(items[0], items[1]):
            return items
        # flip the order
        return items[::-1]

    return build_func


class TestRanking(unittest.TestCase):
    def test_pairwise_ranking_with_missing_values(self) -> None:
        """ """

        num_samples = 100

        true_scores = np.arange(0, num_samples, 1)

        to_be_ranked = true_scores.copy()
        np.random.shuffle(to_be_ranked)

        ranker = PairwiseRanking(
            to_be_ranked,
            compare_all_pairs(
                to_be_ranked,
                partial(pairwise_compare_fn, noise_rate=0.1),
                missing_rate=0.7,
            ),
        )
        ranked_items = ranker.rank(method="BTM")
        sr = spearmanr(ranked_items, true_scores)
        print(f"spearman r = {sr.statistic} p = {sr.pvalue}")

        self.assertTrue(sr.statistic > 0.95)

    def test_efficient_active_ranking_pairwise_model(self) -> None:
        """ """

        num_samples = 10000

        true_scores = np.arange(0, num_samples, 1)

        to_be_ranked = true_scores.copy()
        np.random.shuffle(to_be_ranked)

        ranker = EfficientRanking(
            to_be_ranked, partial(pairwise_compare_fn, noise_rate=0.1), confidence=0.95
        )
        ranked_items = ranker.rank(method="merge")  # or 'quick'
        print(f"Total comparisons: {ranker.total_comparisons}")
        sr = spearmanr(ranked_items, true_scores)
        print(f"spearman r = {sr.statistic} p = {sr.pvalue}")

        self.assertTrue(sr.statistic > 0.89)

    def test_efficient_ranking_batch_rank(self) -> None:
        """ """

        num_samples = 100000
        budget = 200000

        true_scores = np.arange(0, num_samples, 1)
        to_be_ranked = true_scores.copy()
        np.random.shuffle(to_be_ranked)

        ranked_items = aggregate_rankings(
            to_be_ranked,
            partial(list_rank_func, number_of_random_flipped=20),
            budget=budget,
        )

        # print(f"Total comparisons: {ranker.total_comparisons}")
        sr = spearmanr(ranked_items, true_scores)
        print(f"spearman r = {sr.statistic} p = {sr.pvalue}")

        ##############

        ranker_func = convert_pairwise_to_ranker(pairwise_compare_fn)

        true_scores = np.arange(0, num_samples, 1)
        to_be_ranked = true_scores.copy()
        np.random.shuffle(to_be_ranked)

        ranked_items = aggregate_rankings(
            to_be_ranked,
            ranker_func,
            budget=budget * 4,
            ranking_model_rank_batch_size=2,
        )

        # print(f"Total comparisons: {ranker.total_comparisons}")
        sr = spearmanr(ranked_items, true_scores)
        print(f"spearman r = {sr.statistic} p = {sr.pvalue}")


if __name__ == "__main__":
    unittest.main()
