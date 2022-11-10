from lifelines.utils import concordance_index
import numpy as np


class MetricsSurvival:
    @staticmethod
    def c_index(
        pred: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray,
        event_class_index: int = -1,
        time_unit: int = 1,
    ) -> float:
        """
        Compute c-index (concordance index) score using lifelines
        :param pred: prediction array per sample. Each element is a score (scalar). Higher score - higher chance for the event
        :param event_times: event/censor time array
        :param event_observed: Optioal- a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.
        :param event_class_index: Optional - the index of the event class in predicted scores tuples
        :return c-index (concordance index) score
        """
        if isinstance(pred[0], np.ndarray):
            pred = np.asarray(pred)[:, event_class_index]

        event_times = (np.array(event_times) / time_unit).astype(int)

        return concordance_index(event_times, -pred, event_observed)

    @staticmethod
    def expected_cindex(
        pred: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray,
    ) -> float:
        """
        Compute c-index (concordance index) score using lifelines
        :param pred: prediction array per sample. Each element is a score (scalar). Higher score - higher chance for the event
        :param event_times: event/censor time array
        :param event_observed: Optioal- a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.
        :return c-index (concordance index) score
        """
        # converting all to number
        pred = np.array(pred)
        event_times = np.array(event_times)
        event_observed = np.array(event_observed)
        # assuming the calculating the "expected time" until an event
        num_bins = pred.shape[1]
        # [1,...,0] - highest risk
        # [0,...,0,1] - lowest risk
        expected_event_time = (pred * np.arange(num_bins)).sum(axis=1)
        return concordance_index(event_times, expected_event_time, event_observed)


def run_example():
    b = 10000
    bins = 64
    # creating probability distribution across bins
    pred = np.random.random(size=(b, bins))
    pred = pred / pred.sum(axis=1, keepdims=True)
    # creating a "has event" binary variable per example
    event_observed = np.random.random(b) > 0.5
    # creating a list of followup times
    event_times = np.random.randint(low=1, high=100, size=b)

    expected_time_cindex = MetricsSurvival.expected_cindex(
        pred=pred, event_times=event_times, event_observed=event_observed
    )
    # should be close to 0.5, considering it's all random
    print(f"expected c-index is {expected_time_cindex}")


if __name__ == "__main__":
    run_example()
