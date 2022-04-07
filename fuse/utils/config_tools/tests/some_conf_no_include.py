from fuse.utils.config_tools import get_config_function
from typing import List

def load_config(config:dict) -> dict:
    config['banana'] = 123
    return config
