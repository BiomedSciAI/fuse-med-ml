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
from typing import Dict, Optional, Sequence, Union

import numpy as np
import scipy
from statsmodels.stats.contingency_tables import mcnemar


class ModelComparison:
    """
    Methods for model comparison
    """

    @staticmethod
    def bootstrap_margin_superiority(
        name: Optional[str],
        test_values: Sequence[float],
        reference_values: Sequence[float],
        margin: float = 0.0,
        margin_type: str = "abs",
    ) -> Dict[str, float]:
        """
        compare method - to be used in PairedBootstrap.
        Compute p_value for the hypothesis that a tested model is superior to a reference model by a margin of error.
        :param name: the original metric name - not used
        :param test_values: test values to compare to reference values. Each element represent a bootstrap iteration.
        :param reference_values: reference values. Each element represent a bootstrap iteration.
        :param margin: the difference required to consider test result superior to a reference result
        :param margin_type: "abs" - absolute margin, "rel" - relative margin (actual margin will be margin * reference_result)
        :return: results dictionary. format: {"diff": <>, "count": <>, "count_superior": <>, "p_value": <>}
        """
        test_values = np.array(test_values)
        reference_values = np.array(reference_values)

        results = {"count": len(test_values)}
        margin_abs = margin
        if margin_type == "rel":
            margin_abs *= reference_values
        diff = test_values - reference_values
        results["count_superior"] = np.where(diff > margin_abs, 1, 0).sum()
        results["diff"] = diff.sum() / results["count"]
        results["p_value"] = 1 - results["count_superior"] / results["count"]

        return results

    @staticmethod
    def bootstrap_margin_non_inferiority(
        name: Optional[str],
        test_values: Sequence[float],
        reference_values: Sequence[float],
        margin: float = 0.0,
        margin_type: str = "abs",
    ) -> Dict[str, float]:
        """
        compare method - to be used in PairedBootstrap.
        Compute p_value for the hypothesis that a tested model is not inferior to a reference model by a margin of error.
        :param name: the original metric name - not used
        :param test_values: test values to compare to reference values. Each element represent a bootstrap iteration.
        :param reference_values: reference values. Each element represent a bootstrap iteration.
        :param margin: the difference required to consider test result non inferior to a reference result
        :param margin_type: "abs" - absolute margin, "rel" - relative margin (actual margin will be margin * reference_result)
        :return: results dictionary. format: {"diff": <>, "count": <>, "count_non_inferior": <>, "p_value": <>}
        """
        test_values = np.array(test_values)
        reference_values = np.array(reference_values)

        results = {"count": len(test_values)}
        margin_abs = margin
        if margin_type == "rel":
            margin_abs *= reference_values
        diff = test_values - reference_values
        results["count_non_inferior"] = np.where(diff > -margin_abs, 1, 0).sum()
        results["diff"] = diff.sum() / results["count"]
        results["p_value"] = 1 - results["count_non_inferior"] / results["count"]

        return results

    @staticmethod
    def bootstrap_margin_equality(
        name: Optional[str],
        test_values: Sequence[float],
        reference_values: Sequence[float],
        margin: float = 0.0,
        margin_type: str = "abs",
    ) -> Dict[str, float]:
        """
        compare method - to be used in PairedBootstrap.
        Compute p_value for the hypothesis that a tested model is equal to a reference model by a margin of error.
        :param name: the original metric name - not used
        :param test_values: test values to compare to reference values. Each element represent a bootstrap iteration.
        :param reference_values: reference values. Each element represent a bootstrap iteration.
        :param margin: the difference required to consider test result equal by margin to a reference result
        :param margin_type: "abs" - absolute margin, "rel" - relative margin (actual margin will be margin * reference_result)
        :return: results dictionary. format: {"diff": <>, "count": <>, "count_margin_equal": <>, "p_value": <>}
        """
        test_values = np.array(test_values)
        reference_values = np.array(reference_values)

        results = {"count": len(test_values)}
        margin_abs = margin
        if margin_type == "rel":
            margin_abs *= reference_values
        diff = abs(test_values - reference_values)
        results["count_margin_equal"] = np.where(diff < margin_abs, 1, 0).sum()
        results["diff"] = diff.sum() / results["count"]
        results["p_value"] = 1 - results["count_margin_equal"] / results["count"]

        return results

    @staticmethod
    def delong_auc_test(
        pred1: Sequence[np.ndarray],
        pred2: Sequence[np.ndarray],
        target: Sequence[np.ndarray],
        pos_class_index: int = -1,
    ) -> Dict:
        """
        Compute p-value resulting from DeLong's statistical test to compare two binary classifiers' ROC AUCs.
        The p-value represents likelihood for the (null) hypothesis that the two ROC AUCs are similar.
        A small p-value means the models are likely to be different.
        :param pred1: list of prediction arrays per sample for the first classifier. Each element shape is [num_classes].
            for a binary classifier, can also be a list of prediction values.
        :param pred2: list of prediction arrays per sample for the second classifier. Each element shape is [num_classes].
            for a binary classifier, can also be a list of prediction values.
        :param target: target per sample. Each element is an integer in range [0 - num_classes)
        :param pos_class_index: index of the positive class (for one vs. rest).
            If the classifier is binary (no use for one vs. rest), then this parameter should be left as its default (-1).
            Otherwise, it denotes which class index is treated as positive (for one vs. rest)
        :return {'p-value', 'z', 'auc1', 'auc2', 'cov11', 'cov12', 'cov22'},
             z is the Z-score (standard score), the normal distribution value for the relevant random variable
                which in our case depends on the models' AUCs and their variance/covariance (DeLong et al. 1988)
             auc1 is the 1st model's AUC
             auc2 is the 2nd model's AUC
             cov11 is the 11 element of DeLong's covariance matrix (variance of the 1st model's AUC)
             cov12 is the 12 element of DeLong's covariance matrix (equal to the 21 element - symmetric matrix)
             cov22 is the 22 element of DeLong's covariance matrix (variance of the 2nd model's AUC)
        """
        if isinstance(pred1[0], float):
            predictions_1 = np.array(pred1)
            predictions_2 = np.array(pred2)
            ground_truth = np.array(target)
        else:
            if pos_class_index < 0:
                pos_class_index = pred1[0].shape[0] - 1
            predictions_1 = np.asarray(pred1)[:, pos_class_index]
            predictions_2 = np.asarray(pred2)[:, pos_class_index]
            ground_truth = (np.array(target) == pos_class_index) * 1

        positive_labels = ground_truth == 1
        negative_labels = ground_truth == 0
        m = (positive_labels).sum()  # number of positives
        n = (negative_labels).sum()  # number of negatives
        x1 = predictions_1[positive_labels]
        y1 = predictions_1[negative_labels]
        x2 = predictions_2[positive_labels]
        y2 = predictions_2[negative_labels]
        # convert to matrix:
        X1 = x1[:, np.newaxis].repeat(n, 1)
        X2 = x2[:, np.newaxis].repeat(n, 1)
        Y1 = y1[np.newaxis, :].repeat(m, 0)
        Y2 = y2[np.newaxis, :].repeat(m, 0)

        PSI1 = (Y1 < X1) * 1.0
        PSI1[Y1 == X1] = 0.5

        PSI2 = (Y2 < X2) * 1.0
        PSI2[Y2 == X2] = 0.5

        # empirical AUC:
        emp_auc_1 = (1.0 / (m * n)) * PSI1.sum()
        emp_auc_2 = (1.0 / (m * n)) * PSI2.sum()

        # structural components (DeLong et al. 1988):
        V10_1 = (1.0 / n) * PSI1.sum(1)
        V10_2 = (1.0 / n) * PSI2.sum(1)

        V01_1 = (1.0 / m) * PSI1.sum(0)
        V01_2 = (1.0 / m) * PSI2.sum(0)

        # covariance matrix
        S10 = (1.0 / (m - 1)) * np.array(
            [
                [np.linalg.norm(V10_1 - emp_auc_1) ** 2, (V10_1 - emp_auc_1).dot(V10_2 - emp_auc_2)],
                [(V10_2 - emp_auc_2).dot(V10_1 - emp_auc_1), np.linalg.norm(V10_2 - emp_auc_2) ** 2],
            ]
        )

        S01 = (1.0 / (n - 1)) * np.array(
            [
                [np.linalg.norm(V01_1 - emp_auc_1) ** 2, (V01_1 - emp_auc_1).dot(V01_2 - emp_auc_2)],
                [(V01_2 - emp_auc_2).dot(V01_1 - emp_auc_1), np.linalg.norm(V01_2 - emp_auc_2) ** 2],
            ]
        )

        S = (1.0 / m) * S10 + (1.0 / n) * S01

        # z-score:
        z = (emp_auc_1 - emp_auc_2) / ((S[0, 0] + S[1, 1] - 2 * S[0, 1]) ** 0.5 + np.finfo(float).eps)

        # p-value:
        p_value = 2 * scipy.stats.norm.sf(abs(z), loc=0, scale=1)

        results = {
            "p_value": p_value,
            "z": z,
            "auc1": emp_auc_1,
            "auc2": emp_auc_2,
            "cov11": S[0, 0],
            "cov12": S[0, 1],
            "cov22": S[1, 1],
        }

        return results

    @staticmethod
    def contingency_table(var1: Sequence[bool], var2: Sequence[bool]) -> np.ndarray:
        """
        Compute contingency table from two paired variables
        :param var1: first variable [list of 0's and 1's or booleans]
        :param var2: second variable [list of 0's and 1's or booleans]
        :return contingency table [2x2 numpy array]
        """
        var1 = np.array(var1, dtype=bool)
        var2 = np.array(var2, dtype=bool)

        con_tab = np.zeros((2, 2))
        con_tab[0, 0] = (np.logical_and(var1, var2)).sum()
        con_tab[0, 1] = (np.logical_and(var1, ~var2)).sum()
        con_tab[1, 0] = (np.logical_and(~var1, var2)).sum()
        con_tab[1, 1] = (np.logical_and(~var1, ~var2)).sum()

        return con_tab

    @staticmethod
    def mcnemars_test(
        pred1: Union[Sequence, np.ndarray],
        pred2: Union[Sequence, np.ndarray],
        target: Optional[Union[Sequence, np.ndarray]],
        exact: bool = True,
    ) -> Dict:
        """
        Compute p-value resulting from McNemar's statistical test to compare two models' predictions or accuracies.
        The p-value represents likelihood for the (null) hypothesis that the two paired variables are similar
        in the sense that they have a similar proportion of disagreements.
        A small p-value means they are likely to be different.
        :param pred1: predictions of the first model.
        :param pred2: predictions of the second model.
        :param target: ground truth vector
        :param exact: If True, then the binomial distribution will be used. Otherwise, the chi-square distribution, which is the approximation to the distribution of the test statistic for large sample sizes.
            The exact test is recommended for small (<25) number of discordants in the contingency table
        :return {'p-value', 'statistic', 'n1', 'n2'},
             'statistic' is the min(n1, n2), where n1, n2 are cases that are zero
                in one classifier but one in the other. This statistic is used in the exact binomial distribution test.
             'n1' and 'n2' as just defined
        """
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        target = np.array(target)

        if target is not None:
            pred1_correct = pred1 == target
            pred2_correct = pred2 == target
            contingency_table = ModelComparison.contingency_table(pred1_correct, pred2_correct)
        else:
            contingency_table = ModelComparison.contingency_table(pred1, pred2)

        # calculate mcnemar test
        res = mcnemar(contingency_table, exact=exact)

        results = {
            "p_value": res.pvalue,
            "statistic": res.statistic,
            "n1": contingency_table[0, 1],
            "n2": contingency_table[1, 0],
        }

        return results


def psi(x, y):  # Heaviside function
    if y < x:
        return 1
    elif y == x:
        return 0.5
    else:
        return 0
