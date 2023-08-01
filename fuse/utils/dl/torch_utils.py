import torch
import functools


def select(condlist, choicelist, default_val):
    # Mimicking numpy.select https://numpy.org/doc/stable/reference/generated/numpy.select.html
    # by iteratively invoking np.where(condtion, choice, default_val) in reverse order
    args_list = reversed(list(zip(condlist, choicelist)))
    func = lambda d, args: torch.where(*args, other=d)
    return functools.reduce(func, args_list, default_val)
