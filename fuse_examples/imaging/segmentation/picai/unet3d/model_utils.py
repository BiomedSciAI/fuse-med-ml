import importlib


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2**k for k in range(num_levels)]


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")
