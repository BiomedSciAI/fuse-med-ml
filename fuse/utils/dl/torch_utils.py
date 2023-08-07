from typing import List, Union
import torch
import functools


def select(
    condlist: List[torch.Tensor],
    choicelist: List[torch.Tensor],
    default_val: Union[float, int],
) -> torch.Tensor:
    """Mimicking numpy.select https://numpy.org/doc/stable/reference/generated/numpy.select.html
        by iteratively invoking np.where(condtion, choice, default_val) in reverse order
    :param List[torch.Tensor] condlist: The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    :param List[torch.Tensor] choicelist: The list of tensors from which the output elements are taken. It has
        to be of the same length as `condlist`.
    :param  Union[float, int] default_val: The element inserted in `output` when all conditions evaluate to False.
    :return torch.Tensor: The value at each position corresponds to the value in the corresponding position in the m-th choicelist
                          if the first 'True' value in the corresponding positions in condlist tensors is m.
                          When all corresponding positions in the condlist are 'False', the value will 'default_val'
    """
    args_list = reversed(list(zip(condlist, choicelist)))
    func = lambda d, args: torch.where(*args, other=d)
    return functools.reduce(func, args_list, default_val)
