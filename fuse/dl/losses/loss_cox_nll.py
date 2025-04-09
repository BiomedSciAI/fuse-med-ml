import torch
from typing import Union, List, Tuple


def _to_numeric_array(
    x: Union[float, int, List[Union[float, int]], Tuple[Union[float, int]]]
) -> Union[float, List[float]]:
    if isinstance(x, str):
        raise ValueError(f"Invalid input: Expected numeric values but got string '{x}'")
    if isinstance(x, list) or isinstance(x, tuple):
        return [float(i) for i in x]  # Convert each element to float
    return float(x)


class CoxNLL(torch.nn.Module):
    """
    Cox Negative Log-Likelihood (NLL) loss.

    This class implements the Cox Proportional Hazards model's Negative Log-Likelihood loss.
    It is a survival analysis technique used to model the time until an event occurs, with the risk
    of the event depending on the features of the individual. The loss function compares the
    predicted hazards (hazard) with the observed event times (event_time) and event statuses (is_event).

    The Cox NLL loss can be expressed as:

    L = -Σ [ δ_i * ( log(Hazard_i) - log(Σ_j Hazard_j * I(T_j ≥ T_i)) ) ]

    Where:
    - δ_i is the event indicator (1 if event occurred, 0 if censored)
    - Hazard_i is the predicted hazard for sample i
    - T_i is the event time for sample i
    - Σ_j Hazard_j * I(T_j ≥ T_i) is the sum of hazards of all individuals who are at risk at time T_i
    """

    def __init__(
        self, time_unit: int = 1, random_ties: bool = True, epsilon: float = 1e-7
    ):
        """
        Initializes the Cox Negative Log-Likelihood (NLL) loss class.

        Args:
        ----
            time_unit (int, optional): The time unit used for scaling event times. Defaults to 1.
            random_ties (bool, optional): Whether to handle ties randomly. If False, uses Breslow's method for handling ties. Defaults to True.
            epsilon (float, optional): Small constant added to prevent division by zero. Defaults to 1e-7.

        """
        super().__init__()
        self.time_unit = time_unit
        self.random_ties = random_ties
        self.epsilon = epsilon

    def forward(
        self, hazard: torch.Tensor, is_event: torch.Tensor, event_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Cox Negative Log-Likelihood (NLL) loss.

        Args:
        ----
            hazard (torch.Tensor): The predicted hazard values for each sample (e.g., the output of a model).
            is_event (torch.Tensor): A tensor of binary values indicating whether the event occurred (1) or the sample is censored (0).
            event_time (torch.Tensor): A tensor of event times corresponding to each sample.

        Returns:
        -------
            torch.Tensor: The computed Cox NLL loss.
        """
        is_event = torch.as_tensor(_to_numeric_array(is_event), dtype=torch.float32).to(
            hazard.device
        )
        event_time = torch.as_tensor(
            _to_numeric_array(event_time), dtype=torch.float32
        ).to(hazard.device)
        event_time = (event_time / self.time_unit).int()

        hazard = torch.as_tensor(hazard).flatten()
        n_events = is_event.sum().float()

        if self.random_ties:
            # efficient but inaccurate version - solves ties randomly
            last_to_first_sorter = torch.argsort(event_time).flip(0)
            sorted_is_event = is_event[last_to_first_sorter]
            sorted_hazard = hazard[last_to_first_sorter]
            loss_per_sample = sorted_is_event * (
                torch.logcumsumexp(sorted_hazard, 0) - sorted_hazard
            )
        else:
            # the true risk set implementation
            # This is Breslow method for handling ties: events with the same time will have the same risk set ==> same denominator
            risk_set = event_time.unsqueeze(0) >= event_time.unsqueeze(
                1
            )  # risk_set[i,j]:  whether T[j] > T[i]
            logits = torch.where(
                risk_set, hazard, -torch.inf
            )  # logits[i,j] = hazard[j] if T[j] > T[i] else -Inf
            loss_per_sample = is_event.float() * (
                torch.logsumexp(logits, dim=1) - hazard
            )  # loss_per_sample[i] = the loss if E[i]=1 else 0

        # epsilon helps avoiding zero division issues
        loss = loss_per_sample.sum() / (n_events + self.epsilon)

        return loss
