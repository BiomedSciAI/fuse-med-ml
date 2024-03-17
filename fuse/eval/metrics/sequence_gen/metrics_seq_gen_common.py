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
from typing import Optional, Tuple, List
from functools import partial

import torch
import numpy as np

from fuse.eval.metrics.metrics_common import MetricPerBatchDefault


class MetricPerplexity(MetricPerBatchDefault):
    def __init__(
        self,
        preds: str,
        target: str,
        ignore_index: Optional[int] = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            log_probs="log_probs",  # collect log_probs - output of  _perplexity_update
            token_num="token_num",  # collect token_num - output of  _perplexity_update
            metric_per_batch_func=None,
            metric_per_batch_func_pre_collect=partial(
                _perplexity_update,
                ignore_index=ignore_index,
                preds_key=preds,
                targets_key=target,
            ),
            result_aggregate_func=_perplexity_compute,
            **kwargs,
        )


# Copied internal function https://github.com/Lightning-AI/metrics/blob/825d17f32ee0b9a2a8024c89d4a09863d7eb45c3/src/torchmetrics/functional/text/perplexity.py#L68
# copied and not imported to not be affected by internal interface modifications.
def _perplexity_update(
    batch_dict: dict,
    preds_key: str,
    targets_key: str,
    ignore_index: Optional[int] = None,
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
    preds = batch_dict[preds_key]
    target = batch_dict[targets_key]
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds)

    if isinstance(target, np.ndarray):
        target = torch.tensor(target)

    if not isinstance(target, torch.Tensor):
        return {"log_probs": None, "token_num": None}

    assert (
        len(preds.shape) == 3
    ), f"Error: expected num dims is 3, got shape {preds.shape}"
    assert (
        len(target.shape) == 2
    ), f"Error: expected num dims is 2, got shape {target.shape}"
    # to save GPU memory
    preds = preds.detach()
    target = target.detach()

    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)

    if ignore_index is not None:
        mask = target.ne(ignore_index)
        target = target.where(
            target != ignore_index, torch.tensor(0, device=target.device)
        )
    else:
        mask = torch.ones_like(target, dtype=torch.bool)

    preds = preds[:, target].diagonal()[mask]
    # avoid from overflow
    if preds.dtype == torch.float16:
        preds = preds.to(torch.float32)
        preds = torch.clamp(preds, min=1e-10)
    total_log_probs = -preds.log().sum()
    count = mask.sum()

    return {"log_probs": total_log_probs.unsqueeze(0), "token_num": count.unsqueeze(0)}


def _perplexity_compute(
    log_probs: List[np.ndarray],
    token_num: List[np.ndarray],
) -> float:
    # avoid from overflow on large epochs
    log_probs = [e.astype(np.float64) for e in log_probs]
    token_num = [e.astype(np.int64) for e in token_num]

    sum_log_probs = sum(log_probs)
    num_total = sum(token_num)
    return float(np.exp(sum_log_probs / num_total))
