from typing import Sequence, Union, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch import nn, optim
from scipy import stats


class Calibration:
    """
    Methods for probability calibration
    """

    @staticmethod
    def reliability_diagram(
        pred: Sequence[Union[np.ndarray, float]],
        target: Sequence[Union[np.ndarray, int]],
        num_bins: Optional[int] = 10,
        num_quantiles: Optional[int] = None,
        output_filename: Optional[str] = None,
    ) -> Dict:
        """
        Reliability diagram
        :param pred: List of arrays of shape [NUM_CLASSES]
        :param target: List of integers [0, NUM_CLASSES) denoting ground truth classes
        :num_bins: Integer denoting the number of equal width bins in the diagram's x-axis.
        :num_quantiles: Integer denoting the number of equal number of samples quantiles in the diagram's x-axis.
            Note, either num_bins or num_quantiles should be set to None, as they define different ways to split the x-axis.
        :output_filename: Path to file to which to save the plot
        :return: saving roc curve to a file and return the input to figure to a dictionary
        """
        assert num_bins is None or num_quantiles is None

        if isinstance(pred[0], float):  # binary case
            pred = [np.array([1 - p, p]) for p in pred]

        max_pred = [np.max(p) for p in pred]
        cls_pred = [np.argmax(p) for p in pred]

        assert all(
            [math.isclose(p.sum(), 1, rel_tol=1e-5) for p in pred]
        )  # make sure that the inputs make proper probabilities

        total_samples = len(pred)
        max_pred = np.array(max_pred)
        if num_bins is not None:
            conf_vec = np.linspace(0, 1, num_bins + 1)
        else:
            quantiles_vec = np.linspace(0, 1, num_quantiles + 1)
            conf_vec = stats.mstats.mquantiles(max_pred, quantiles_vec.tolist())
        acc_vec = np.zeros(len(conf_vec) - 1)
        fraction_of_samples = np.zeros(len(conf_vec) - 1)
        num_samples = np.zeros(len(conf_vec) - 1)
        for i in range(len(conf_vec) - 1):
            cnt_correct_in_conf = sum(
                [
                    p_cls == t
                    for (p_cls, p, t) in zip(cls_pred, max_pred, target)
                    if p >= conf_vec[i] and p < conf_vec[i + 1]
                ]
            )
            cnt_conf = len([p for p in max_pred if p >= conf_vec[i] and p < conf_vec[i + 1]])
            num_samples[i] = cnt_conf
            fraction_of_samples[i] = cnt_conf / np.max([total_samples, np.finfo(float).eps])
            acc_vec[i] = cnt_correct_in_conf / np.max([cnt_conf, np.finfo(float).eps])
        mid_conf_vec = np.diff(conf_vec) / 2 + conf_vec[:-1]
        cnt_correct_total = sum([p_cls == t for (p_cls, t) in zip(cls_pred, target)])
        avg_acc = cnt_correct_total / total_samples
        avg_conf = np.mean(max_pred)
        # extract info for the plot
        results = {}
        results["conf_vec"] = conf_vec
        results["mid_confidence_points"] = mid_conf_vec
        results["accuracy"] = acc_vec
        results["avg_accuracy"] = avg_acc
        results["avg_confidence"] = avg_conf
        results["num_samples_per_bin"] = num_samples
        results["total_samples"] = total_samples

        # display
        if output_filename is not None:
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
            f.tight_layout(pad=3.0)
            ax1.plot(mid_conf_vec, fraction_of_samples, color="blue", marker="o", linewidth=2)
            ax1.set_title(
                f"Confidence Histogram. Avg. Accuracy: {np.round(avg_acc, 2)}, Avg. Confidence: {np.round(avg_conf, 2)}"
            )
            ax1.set_xlabel("Confidence")
            ax1.set_ylabel("Fraction of samples")
            ax1.grid()
            ax2.plot(conf_vec, conf_vec, color="black", linewidth=2)
            ax2.plot(mid_conf_vec, acc_vec, color="red", marker="o", linewidth=2)
            ax2.set_title("Reliability Curve")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Accuracy")
            ax2.grid()
            f.savefig(output_filename)
            plt.close()

        return results

    @staticmethod
    def ece(
        pred: Sequence[Union[np.ndarray, float]],
        target: Sequence[Union[np.ndarray, int]],
        num_bins: Optional[int] = 10,
        num_quantiles: Optional[int] = None,
    ) -> float:
        """
        :param pred: prediction array per sample. Each element shape [num_classes]
        :param target: target per sample. Each element is an integer in range [0 - num_classes)
        :num_bins: Integer denoting the number of equal width bins in the diagram's x-axis.
        :num_quantiles: Integer denoting the number of equal number of samples quantiles in the diagram's x-axis.
            Note, either num_bins or num_quantiles should be set to None, as they define different ways to split the x-axis.
        :return Expected Calibration Error (ECE) score
        """
        reliability_results = Calibration.reliability_diagram(
            pred, target, num_bins=num_bins, num_quantiles=num_quantiles, output_filename=None
        )
        conf_vec = reliability_results["conf_vec"][1:]
        total_samples = reliability_results["total_samples"]
        num_samples_per_bin = reliability_results["num_samples_per_bin"]
        existing_bins = num_samples_per_bin > 0
        accuracy_vec = reliability_results["accuracy"]
        # expected calibration error
        ece = (1.0 / total_samples) * np.sum(num_samples_per_bin * np.abs(accuracy_vec - conf_vec))
        # maximum calibration error
        # we filter out bins which don't contain any samples
        mce = np.max(np.abs(accuracy_vec[existing_bins] - conf_vec[existing_bins]))

        results = {}
        results["ece"] = ece
        results["mce"] = mce

        return results

    @staticmethod
    def find_temperature(pred: Sequence[Union[np.ndarray, float]], target: Sequence[Union[np.ndarray, int]]) -> float:
        """
        :param pred: prediction array per sample. Each element shape [num_classes]
            note that pred should be logits *before* conversion to softmax probabilities
        :param target: target per sample. Each element is an integer in range [0 - num_classes)
        :return temperature scale for calibration
        We optimize to find the temperature similarly to: https://github.com/gpleiss/temperature_scaling
        """
        nll_criterion = nn.CrossEntropyLoss()
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
        with torch.no_grad():
            logits = torch.tensor(np.array(pred))
            target = torch.tensor(target)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / temperature, target)
            loss.backward()
            return loss

        optimizer.step(eval)

        return temperature.item()

    @staticmethod
    def apply_temperature(pred: Sequence[Union[np.ndarray, float]], temperature: Union[float, None] = None) -> float:
        """
        :param pred: prediction array per sample. Each element shape [num_classes]
            note that pred should be logits *before* conversion to softmax probabilities
        :param temperature: temperature scalar
        :return array of softmax probabilities
        """
        if temperature is None:
            pred_calibrated = pred
        else:
            if isinstance(pred[0], np.ndarray):
                pred = np.array(pred)
            pred_calibrated = torch.tensor(pred / temperature)
            pred_calibrated = nn.Softmax(dim=1)(pred_calibrated)

        return np.array(pred_calibrated)
