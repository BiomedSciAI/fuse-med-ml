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
import pandas as pd
from typing import Any, Dict
from collections import OrderedDict

from fuse.eval.metrics.sequence_gen.metrics_seq_gen_common import (
    MetricPerplexity,
    MetricCountSeqAndTokens,
)

from fuse.eval.evaluator import EvaluatorDefault


from fuse.data import PipelineDefault, DatasetDefault, OpReadDataframe, CollateDefault
import torch
from torch.utils.data.dataloader import DataLoader


def example_seq_gen_0(seed: int = 1234) -> Dict[str, Any]:
    """
    Example/Test for perplexity metric
    """
    # entire dataframe at once case
    torch.manual_seed(seed=seed)
    pred = torch.randn(1000, 7, 100)
    pred = torch.nn.functional.softmax(pred, dim=-1).numpy()
    target = torch.randint(0, 100, (1000, 7)).numpy()

    data = {
        "pred": list(pred),
        "label": list(target),
        "id": list(range(1000)),
    }
    data = pd.DataFrame(data)

    metrics = OrderedDict(
        [
            (
                "perplexity",
                MetricPerplexity(preds="pred", target="label"),
            )
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data, metrics=metrics)
    print(results)

    return results


def example_seq_gen_1(seed: int = 1234) -> Dict[str, Any]:
    """
    Example/Test for perplexity metric - batch mode
    """
    # entire dataframe at once case
    torch.manual_seed(seed=seed)
    pred = torch.randn(1000, 7, 100)
    pred = torch.nn.functional.softmax(pred, dim=-1).numpy()
    target = torch.randint(0, 100, (1000, 7)).numpy()

    data = {
        "pred": list(pred),
        "label": list(target),
        "id": list(range(1000)),
    }
    data = pd.DataFrame(data)

    # Working with pytorch dataloader mode
    dynamic_pipeline = PipelineDefault(
        "test",
        [
            (OpReadDataframe(data, key_column="id"), dict()),
        ],
    )
    ds = DatasetDefault(sample_ids=len(data), dynamic_pipeline=dynamic_pipeline)
    ds.create()
    dl = DataLoader(ds, collate_fn=CollateDefault(), batch_size=100)
    metrics = OrderedDict(
        [("perplexity", MetricPerplexity(preds="pred", target="label"))]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=dl, metrics=metrics, batch_size=0)
    print(results)

    return results


def example_seq_gen_2() -> Dict[str, Any]:
    """
    Example/Test for perplexity metric - batch mode
    """

    encoder_input_tokens = torch.arange(5000).reshape(10, 500)
    data = {
        "encoder_input_tokens": list(encoder_input_tokens),
        "id": list(range(10)),
    }
    data = pd.DataFrame(data)

    # Working with pytorch dataloader mode
    dynamic_pipeline = PipelineDefault(
        "test",
        [
            (OpReadDataframe(data, key_column="id"), dict()),
        ],
    )
    ds = DatasetDefault(sample_ids=len(data), dynamic_pipeline=dynamic_pipeline)
    ds.create()
    dl = DataLoader(ds, collate_fn=CollateDefault())
    metrics = OrderedDict(
        [
            (
                "count",
                MetricCountSeqAndTokens(
                    encoder_input="encoder_input_tokens", ignore_index=4999
                ),
            )
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=dl, metrics=metrics, batch_size=0)
    print(results)

    return results
