"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

from typing import List, Dict, Any, Optional
from os.path import join
from yaml import safe_load
from os import environ
from .configmodels import EHRTransformerConfigSettings


def get_config(
    fname: Optional[str] = None,
):
    if fname is None:
        config = EHRTransformerConfigSettings()
    else:
        with open(fname, encoding="utf8") as file:
            yaml_config = safe_load(file)
        config = EHRTransformerConfigSettings(**yaml_config)

    global_dict = make_global_dict(config.data.dict(), config.global_settings.dict(), config.learning.dict())
    base_path = global_dict["base_path"]
    return (
        global_dict,
        make_file_config_dict(base_path, config.data.task),
        config.model.dict(),
        config.learning.optimization.dict(),
        config.data.dict(),
        config.naming_conventions.dict(),
    )


def get_config_raw(fname: str) -> List[Dict[str, Any]]:
    """Read config from yaml

    Reads a yaml file with config options and outputs them in the same format
    as the legacy config code.

    Args:
        fname (str): path to yaml file

    Returns:
        List[Dict[str, Any]]: dicts matching `global_params`, `file_config`,
            `model_config`, `optim_config`, `data_config`, `naming_config`
    """
    with open(fname, encoding="utf8") as file:
        yaml_config = safe_load(file)

    data_dict = yaml_config["data"]
    global_dict = yaml_config["global"]
    learning_dict = yaml_config["learning"]
    model_dict = yaml_config["model"]
    optimization_dict = yaml_config["learning"]["optimization"]
    naming_dict = yaml_config["naming_conventions"]
    global_dict = make_global_dict(data_dict, global_dict, learning_dict)

    return (
        global_dict,
        make_file_config_dict(global_dict["base_path"], data_dict["task"]),
        model_dict,
        optimization_dict,
        data_dict,
        naming_dict,
    )


def make_global_dict(data_dict, global_dict, learning_dict):
    global_update_dict = _get_global_update_dict(**data_dict, **global_dict)

    global_update_dict.update({k: v for k, v in learning_dict.items() if not k == "optimization"})
    return {**global_dict, **global_update_dict}


def _get_global_update_dict(
    uber_base_path: str,
    task: str,
    subtask: str,
    out_type: str,
    task_type: str,
    output_name: str,
    limit_visits: str = None,
    days_to_inddate: str = None,
    days_to_inddate_tr: str = None,
    event_prediction_window_days: str = None,
    event_prediction_window_days_tr: str = None,
    use_procedures: bool = False,
    visit_days_resolution: str = None,
    **kwargs,
) -> Dict[str, str]:
    """Use variables from supplied config to generate config options for
    downstream components.

    The kwargs arg is in place to enable dumping the incoming settings dicts.

    Args:
        uber_base_path (str): the basest base path
        task (str): the task name (aka cohort name)
        subtask (str): the subtask name
        out_type (str): the output type
        task_type (str): the task type
        output_name (str): the output name such as `admdate`

    Returns:
        Dict[str, str]: a dict to update the global section
    """
    task_subdir = f"{task}_{subtask}"
    if limit_visits is not None:
        task_subdir += f"_{limit_visits}"
    if days_to_inddate is not None:
        if int(days_to_inddate) >= 0:
            task_subdir += f"_{days_to_inddate}_to_ind"
        else:
            task_subdir += f"_{abs(int(days_to_inddate))}_after_ind"
    if days_to_inddate_tr is not None:
        if int(days_to_inddate_tr) >= 0:
            task_subdir += f"_tr_{days_to_inddate_tr}_to_ind"
        else:
            task_subdir += f"_tr_{abs(int(days_to_inddate_tr))}_after_ind"
    if event_prediction_window_days is not None:
        if int(event_prediction_window_days) >= 0:
            task_subdir += f"_{event_prediction_window_days}_to_event"
        else:
            task_subdir += f"_{abs(int(event_prediction_window_days))}_after_event"
    if event_prediction_window_days_tr is not None:
        if int(event_prediction_window_days_tr) >= 0:
            task_subdir += f"_tr_{event_prediction_window_days_tr}_to_event"
        else:
            task_subdir += f"_tr_{abs(int(event_prediction_window_days_tr))}_after_event"
    if visit_days_resolution is not None:
        if int(visit_days_resolution) > 2:
            task_subdir += f"_merge_{visit_days_resolution}_days"
    if use_procedures:
        task_subdir += "_with_procedures"

    base_path = join(uber_base_path, task_subdir, out_type)
    output_dir_main = join(base_path, "models", task, "_".join([task_type, output_name]))
    best_name = "_".join([task_type, output_name, "best"])
    last_name = "_".join([task_type, output_name, "last"])
    global_update_dict = {
        "best_name": best_name,
        "last_name": last_name,
        "output_dir_main": output_dir_main,
        "output_dir": None,
        "base_path": base_path,
    }

    return global_update_dict


def make_file_config_dict(base_path: str, task: str) -> Dict[str, str]:
    """Generate file config dict.

    Args:
        base_path (str): base path
        task (str): task name

    Returns:
        Dict[str,str]: the file config dict with keys "vocab", "train" and
            "test"
    """
    return {
        # vocab token2idx idx2token
        "vocab": join(base_path, f"{task}_small.vocab"),
        "gt_vocab": join(base_path, f"{task}_gt_small.vocab"),
        "train": join(base_path, "train", task),
        "test": join(base_path, "test", task),
        "val": join(base_path, "val", task),
    }


def get_vocab_path(root_vocab_path: str, event_type_identifier: str = ""):
    return root_vocab_path + event_type_identifier
