import functools


def func_compose_2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def func_compose(*fs):
    return functools.reduce(func_compose_2, fs)
