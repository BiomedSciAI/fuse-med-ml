from functools import partial

from fuse.eval.metrics.libs.calibration import Calibration

from .metrics_classification_common import MetricMultiClassDefault


class MetricReliabilityDiagram(MetricMultiClassDefault):
    """
    Compute and save a plot of the Reliability Diagram, a plot depicting a model's
    accuracy vs. confidence scores. For a well calibrated model, the result should be the identity function.
    See: https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(
        self,
        pred: str,
        target: str,
        num_bins: int | None = 10,
        num_quantiles: int | None = None,
        output_filename: str | None = None,
        **kwargs: dict
    ) -> None:
        """
        :param pred: key name for the model prediction scores
        :param target: key name for the ground truth target values
        :param num_bins: number of equal width bins in the diagram's x-axis.
        :param num_quantiles: alternatively, number of equal number of samples quantiles in the diagram's x-axis.
        :param output_filename: output filename to save the figure. if None, will just save the arguments to create the figure in results dictionary.
        """
        reliability_diagram = partial(
            Calibration.reliability_diagram,
            num_bins=num_bins,
            num_quantiles=num_quantiles,
            output_filename=output_filename,
        )
        super().__init__(
            pred=pred, target=target, metric_func=reliability_diagram, **kwargs
        )


class MetricECE(MetricMultiClassDefault):
    """
    Compute the Expected Calibration Error, a measure for a model output
    probability score miscalibration (deviation from true probability for correctness).
    See: https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(
        self,
        pred: str,
        target: str,
        num_bins: int | None = 10,
        num_quantiles: int | None = None,
        **kwargs: dict
    ):
        """
        :param pred: key name for the model prediction scores
        :param target: key name for the ground truth target values
        :param class_names: class names. required for multi-class classifiers
        """
        ece = partial(Calibration.ece, num_bins=num_bins, num_quantiles=num_quantiles)
        super().__init__(pred=pred, target=target, metric_func=ece, **kwargs)


class MetricFindTemperature(MetricMultiClassDefault):
    """
    Find the optimal "temperature" for calibrating the prediction logits
    to represent calibrated probabilities.
    See: https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(self, pred: str, target: str, **kwargs: dict):
        super().__init__(
            pred=pred, target=target, metric_func=Calibration.find_temperature, **kwargs
        )
        """
        :param pred: key name for the model prediction scores
        :param target: key name for the ground truth target values
        """


class MetricApplyTemperature(MetricMultiClassDefault):
    """
    Apply temperature for calibrating the prediction logits
    to represent calibrated probabilities.
    See: https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(
        self, pred: str, temperature: float | str | None = None, **kwargs: dict
    ):
        super().__init__(
            pred=pred,
            target=None,
            temperature=temperature,
            metric_func=Calibration.apply_temperature,
            **kwargs
        )
        """
        :param pred: key name for the model prediction scores
        :param target: key name for the ground truth target values
        """
