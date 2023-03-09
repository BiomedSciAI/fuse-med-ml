"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""
from typing import Optional, Tuple, List, Union
import torch

from fuse.eval.metrics.metrics_common import MetricPerBatchDefault
import numpy as np


class MetricPerplexity(MetricPerBatchDefault):
    def __init__(self, preds: str, target: str, **kwargs) -> None:
        super().__init__(
            preds=preds,
            target=target,
            metric_per_batch_func=_perplexity_update,
            result_aggregate_func=_perplexity_compute,
            post_keys_to_collect=["log_probs", "token_num"],
            **kwargs
        )


# Copied internal function https://github.com/Lightning-AI/metrics/blob/825d17f32ee0b9a2a8024c89d4a09863d7eb45c3/src/torchmetrics/functional/text/perplexity.py#L68
# copied and not imported to not be affected by internal interface modifications.
def _perplexity_update(
    preds: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray], ignore_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute intermediate statistics for Perplexity.
    Args:
        preds:
            Probabilities scores (after softmax) assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.
    Returns:
        Log probabilities, summed over all samples
        Number of tokens
    """
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds)

    if isinstance(target, np.ndarray):
        target = torch.tensor(target)

    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)

    if ignore_index is not None:
        mask = target.ne(ignore_index)
        target = target.where(target != ignore_index, torch.tensor(0, device=target.device))
    else:
        mask = torch.ones_like(target, dtype=torch.bool)

    preds = preds[:, target].diagonal()[mask]
    total_log_probs = -preds.log().sum()
    count = mask.sum()

    return {"log_probs": total_log_probs.unsqueeze(0), "token_num": count.unsqueeze(0)}


def _perplexity_compute(
    log_probs: List[np.ndarray],
    token_num: List[np.ndarray],
) -> float:
    sum_log_probs = sum(log_probs)
    num_total = sum(token_num)
    return float(np.exp(sum_log_probs / num_total))
