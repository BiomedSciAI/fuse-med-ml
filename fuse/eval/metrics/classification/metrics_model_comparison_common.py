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

from typing import Optional, Sequence

from .metrics_classification_common import MetricMultiClassDefault
from fuse.eval.metrics.libs.model_comparison import ModelComparison
from fuse.eval.metrics.metrics_common import MetricDefault


class MetricDelongsTest(MetricMultiClassDefault):
    def __init__(self, pred1: str, pred2: str, target: str, class_names: Optional[Sequence[str]] = None, **kwargs):
        # :param pred1: key name for the predictions of model 1
        # :param pred2: key name for the predictions of model 2
        # :param target: key name for the ground truth labels
        # :param class_names: class names. required for multi-class classifiers

        super().__init__(
            pred=None,
            target=target,
            metric_func=ModelComparison.delong_auc_test,
            class_names=class_names,
            pred1=pred1,
            pred2=pred2,
            **kwargs
        )


class MetricContingencyTable(MetricDefault):
    def __init__(self, var1: str, var2: str, **kwargs):
        """
        Create contingency table from two paired variables.
        :param var1: key name for the first variable
        :param var2: key name for the second variable
        """
        super().__init__(
            pred=None, target=None, metric_func=ModelComparison.contingency_table, var1=var1, var2=var2, **kwargs
        )


class MetricMcnemarsTest(MetricMultiClassDefault):
    def __init__(self, pred1: str, pred2: str, target: Optional[str] = None, exact: Optional[bool] = True, **kwargs):
        """
        McNemar's statistical test for comparing two model's predictions or accuracies
        in the sense of the statistics of their disagreements, as seen in the contingency table.
        For comparing two classifiers, it's possible to provide the ground truth and then the comparison
        will be of the models' accuracies. It's also possible to perform significance test on the predictions only.
        This can be useful either for classifiers, or any other paired data, even in cases where ground truth
        may not be applicable/available (i.e patients' opinion surveys, unsupervised model predictions, etc'.).
        :param pred1: key name for the class predictions of the first model
        :param pred2: key name for the class predictions of the second model
        :param target: (optional) key name for the ground truth
        :param exact: If True, then the binomial distribution will be used. Otherwise, the chi-square distribution, which is the approximation to the distribution of the test statistic for large sample sizes.
            The exact test is recommended for small (<25) number of discordants in the contingency table
        """
        super().__init__(
            pred=None,
            pred1=pred1,
            pred2=pred2,
            target=target,
            exact=exact,
            metric_func=ModelComparison.mcnemars_test,
            **kwargs
        )
