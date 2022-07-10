import os
from fuse.utils.config_tools import get_config_function


def load_config(config: dict) -> dict:
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _base_conf = os.path.join(_curr_dir, "base_conf_example.py")
    config = get_config_function(_base_conf)(config)
    config["banana"] = 123
    return config
