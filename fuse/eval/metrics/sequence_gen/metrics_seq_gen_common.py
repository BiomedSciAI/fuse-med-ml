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
from typing import Optional, Tuple, List, Dict
from functools import partial
from copy import copy
import torch
import numpy as np

from fuse.eval.metrics.metrics_common import MetricPerBatchDefault


class MetricCountSeqAndTokens(MetricPerBatchDefault):
    """
    Counts the total number sequences and tokens in encoder_input
    """

    def __init__(
        self,
        encoder_input: str,
        ignore_index: Optional[int] = None,
        state: Optional[dict] = None,
        **kwargs: dict,
    ) -> None:
        """
        :param encoder_input: key to the encoder_input
        :param ignore_index: token_id to ignore (not to count), typically pad token id
        :param state: the sequence count and token count to continue for. Should be restored when we continue training.
                    use get_state() to get the state and save it upon checkpointing,
        :param kwargs: additional super class arguments
        """
        super().__init__(
            seq_num="seq_num",  # collect seq_num - output of  _count_seq_and_tokens_update
            token_num="token_num",  # collect token_num - output of  _count_seq_and_tokens_update
            metric_per_batch_func=None,
            metric_per_batch_func_pre_collect=partial(
                _count_seq_and_tokens_update,
                ignore_index=ignore_index,
                encoder_input_key=encoder_input,
            ),
            result_aggregate_func=self._count_seq_and_tokens_compute,
            **kwargs,
        )
        if state is None:
            self._state = {"seq_num": 0, "token_num": 0}
        else:
            assert "seq_num" in state
            assert "token_num" in state
            self._state = state

    def _count_seq_and_tokens_compute(
        self,
        seq_num: List[np.ndarray],
        token_num: List[np.ndarray],
    ) -> dict:

        seq_num_total = sum(seq_num)
        token_num_total = sum(token_num)
        self._state["seq_num"] += seq_num_total
        self._state["token_num"] += token_num_total
        return copy(self._state)

    def get_state(self) -> dict:
        return copy(self._state)


def _count_seq_and_tokens_update(
    batch_dict: dict,
    encoder_input_key: str,
    ignore_index: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Count number of sequences and tokens
    Args:
        encoder_input_key:
            key to encoder_input
        ignore_index:
            Token not to count, typically padding
    Returns:
        dictionary with number of sequences and tokens
    """
    encoder_input = batch_dict[encoder_input_key]

    # to save GPU memory
    encoder_input = encoder_input.detach()

    if ignore_index is not None:
        mask = encoder_input.ne(ignore_index)
    else:
        mask = torch.ones_like(encoder_input, dtype=torch.bool)

    seq_num = torch.tensor(
        mask.shape[0], dtype=torch.int64, device=encoder_input.device
    )
    token_num = mask.sum().to(dtype=torch.int64)

    return {"seq_num": seq_num.unsqueeze(0), "token_num": token_num.unsqueeze(0)}


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
            collect_ids=False,
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
