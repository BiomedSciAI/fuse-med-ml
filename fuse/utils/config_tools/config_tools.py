import runpy
from typing import List, Union, Callable, Dict


class Config:
    def __init__(self):
        pass

    def load(self, *configs: List[Union[Dict, str, Callable]]) -> Dict:
        """
        loads, in order, configs
        each item can be either:
        1. Python dict. the python dict "accumulated" so for will be dict.update-ed with this element dict
        2. path to a python script file. It will be expected to contain a callable:
            def get_config(config:dict) -> dict
                ...
        3. a callable with the signature (config:dict) -> dict
        """
        ans = dict()
        for idx, conf in enumerate(configs):
            if isinstance(conf, dict):
                ans.update(conf)
            elif isinstance(conf, str):
                func = get_config_function(conf)
                ans = func(ans)
            elif callable(conf):
                ans = conf(ans)
            else:
                raise Exception(f"config element #{idx} is not a valid type. It has type={type(conf)}")
        return ans


def get_config_function(script_path: str):
    ans = runpy.run_path(script_path)
    func_name = "load_config"
    if func_name not in ans:
        raise Exception(f"Expected to have load_config(conf:dict) -> dict defined in {script_path}")
    return ans[func_name]
