import random
from typing import List, Callable, Any, Optional
import numpy as np
from tqdm import trange
from collections import defaultdict


def aggregate_rankings(
    all_items: List[Any],
    ranking_model: Callable[[List[Any]], List[Any]],
    budget: Optional[int] = None,
) -> List[Any]:
    """
    Aggregate rankings by efficient pairwise comparison.

    Args:
    - ranking_model: Function that returns a subset ranking
    - all_items: Complete list of items to be ranked
    - num_samples: Number of random sampling iterations

    Returns:
    Globally optimized ranking of items
    """
    total_num_samples = len(all_items)
    if budget is None:
        budget = int(np.ceil(np.log10(total_num_samples) * total_num_samples * 2))
        print("no budget selected, defaulting to budget=", budget)
    else:
        print("budget=", budget)
    item_scores = defaultdict(int)

    # Precompute total unique pairs to avoid redundant comparisons
    unique_pairs = set()

    # Generate pairwise comparisons through random sampling
    for _ in trange(budget):
        # Randomly select a subset of items
        sample_size = min(len(all_items), random.randint(2, 10))
        sample = np.random.choice(all_items, size=sample_size)

        # Get model's ranking for this subset
        ranked_sample = ranking_model(sample)

        # Record pairwise comparisons
        for i in range(len(ranked_sample)):
            for j in range(i + 1, len(ranked_sample)):
                higher_item = ranked_sample[i]
                lower_item = ranked_sample[j]

                # Create a hashable pair
                pair = (higher_item, lower_item)

                # Avoid redundant comparisons
                if pair not in unique_pairs:
                    item_scores[higher_item] += 1
                    item_scores[lower_item] -= 1
                    unique_pairs.add(pair)

    # Sort items based on their aggregate score
    global_ranking = sorted(all_items, key=lambda x: item_scores[x], reverse=True)

    return global_ranking


# Example usage
def example_unreliable_model(items):
    """
    Simulate an inconsistent ranking model
    Randomly ranks a subset of input items
    """
    return random.sample(items, len(items))


# if __name__ == "__main__":
#     items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
#     best_ranking = aggregate_rankings(example_unreliable_model, items)
#     print("Aggregated Global Ranking:", best_ranking)


if __name__ == "__main__":
    from scipy.stats import spearmanr
    from functools import partial

    def compare_fn(items, number_of_random_flipped: int = 0):
        # Your comparison logic to rank the given list of items
        # return sorted_items
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

    num_samples = 100000

    true_scores = np.arange(0, num_samples, 1)

    to_be_ranked = true_scores.copy()
    np.random.shuffle(to_be_ranked)

    ranked_items = aggregate_rankings(
        to_be_ranked,
        partial(compare_fn, number_of_random_flipped=10),
        budget=1000000 * 3,
    )

    # print(f"Total comparisons: {ranker.total_comparisons}")
    sr = spearmanr(ranked_items, true_scores)
    print(f"spearman r = {sr.statistic} p = {sr.pvalue}")
