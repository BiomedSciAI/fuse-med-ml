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

Created on Nov 30, 2023

"""

from fuse.eval.metrics.regression.metrics import MetricPearsonCorrelation
import numpy as np
import pandas as pd
from collections import OrderedDict
from fuse.eval.evaluator import EvaluatorDefault


def example_pearson_correlation() -> float:
    """
    Pearson correlation coefficient
    """

    # define data
    sz = 1000
    data = {
        "id": range(sz),
    }
    np.random.seed(0)
    rand_vec = np.random.randn((sz))
    data["x1"] = 100 * np.ones((sz)) + 10 * rand_vec
    data["x2"] = -10 * np.ones((sz)) + 3 * rand_vec

    data_df = pd.DataFrame(data)

    # list of metrics
    metrics = OrderedDict(
        [
            ("pearsonr", MetricPearsonCorrelation(pred="x1", target="x2")),
        ]
    )

    # read files
    evaluator = EvaluatorDefault()
    res = evaluator.eval(ids=None, data=data_df, metrics=metrics)

    return res
