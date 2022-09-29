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

import logging
import os

from typing import Optional, Union
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuseimg.datasets.knight import KNIGHT

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import save_dataframe

from examples.fuse_examples.imaging.classification.knight.eval.eval import TASK1_CLASS_NAMES, TASK2_CLASS_NAMES
from fuse.dl.lightning.pl_module import LightningModuleDefault
import pytorch_lightning as pl
from examples.fuse_examples.imaging.classification.knight.baseline.fuse_baseline import make_model


def make_predictions_file(
    model_dir: str,
    model: torch.nn.Module,
    checkpoint: str,
    data_path: str,
    cache_path: Optional[str],
    split: Union[str, dict],
    output_filename: str,
    predictions_key_name: str,
    task_num: int,
    auto_select_gpus: Optional[bool] = True,
    reset_cache: bool = False,
):
    """
    Automaitically make prediction files in the requested format - given path to model dir create by FuseMedML during training
    :param model_dir: path to model dir create by FuseMedML during training
    :param model: definition of the model
    :param checkpoint: path to the model checkpoint file
    :param data_path: path to the original data downloaded from https://github.com/neheller/KNIGHT
    :param cache_path: Optional - path to the cache folder. If none, it will pre-processes the data again
    :param split: either path to pickled dictionary or the actual dictionary specifing the split between train and validation. the dictionary maps "train" to list of sample descriptors and "val" to list of sample descriptions
    :param output_filename: filename of the output csv file
    :param predictions_key_name: the key in batch_dict of the model predictions
    :param task_num: either 1 or 2 (task 1 or task 2)
    :param auto_select_gpus: whether to allow lightning to select gpus automatically
    :param reset_cache: whether to reset the cache
    """
    # Logger
    fuse_logger_start(console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("KNIGHT: make predictions file in FuseMedML", {"attrs": ["bold", "underline"]})
    lgr.info(f"predictions_filename={os.path.abspath(output_filename)}", {"color": "magenta"})

    # Data
    # read train/val splits file.
    if isinstance(split, str):
        split = pd.read_pickle(split)
        if isinstance(split, list):
            # For this example, we use split 0 out of the the available cross validation splits
            split = split[0]
    if split is None:  # test mode
        json_filepath = os.path.join(data_path, "features.json")
        data = pd.read_json(json_filepath)
        split = {"test": list(data.case_id)}

    dataset = KNIGHT.dataset(data_path, cache_path, split, reset_cache=reset_cache)
    if type(dataset) == tuple and len(dataset) == 2:
        dataset = dataset[1]
    dl = DataLoader(
        dataset=dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=2,
        num_workers=8,
        collate_fn=CollateDefault(),
    )

    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
    )

    pl_module.set_predictions_keys([predictions_key_name])

    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        accelerator="gpu",
        devices=1,
        strategy=None,
        auto_select_gpus=auto_select_gpus,
    )

    predictions = pl_trainer.predict(pl_module, dl, ckpt_path=checkpoint)

    # Convert to required format
    if task_num == 1:
        class_names = TASK1_CLASS_NAMES
    elif task_num == 2:
        class_names = TASK2_CLASS_NAMES
    else:
        raise Exception(f"Unexpected task num {task_num}")

    predictions_score_names = [f"{cls_name}-score" for cls_name in class_names]
    data = []
    for prediction in predictions:
        for i in range(len(prediction["id"])):
            data.append(list(prediction["model.output.head_0"][i]))

    predictions_df = pd.DataFrame(data, columns=list(predictions_score_names))
    predictions_df.reset_index(inplace=True)
    predictions_df.rename({"index": "case_id"}, axis=1, inplace=True)

    # save file
    save_dataframe(predictions_df, output_filename, index=False)


if __name__ == "__main__":
    """
    Automaitically make prediction files in the requested format - given model definition and path to model dir create by FuseMedML during training
    """
    # no arguments - set arguments inline - see details in function make_predictions_file
    model_dir = ""
    checkpoint = "best"
    data_path = ""
    cache_path = ""
    split = "baseline/splits_final.pkl"
    output_filename = "validation_predictions.csv"
    predictions_key_name = "model.output.head_0"
    task_num = 1  # 1 or 2

    use_data = {"imaging": True, "clinical": True}  # specify whether to use imaging, clinical data or both
    num_classes = 2
    imaging_dropout = 0.5
    clinical_dropout = 0.0
    fused_dropout = 0.5

    model = make_model(
        use_data=use_data, num_classes=num_classes, imaging_dropout=imaging_dropout, fused_dropout=fused_dropout
    )

    make_predictions_file(
        model_dir=model_dir,
        model=model,
        checkpoint=checkpoint,
        data_path=data_path,
        cache_path=cache_path,
        split=split,
        output_filename=output_filename,
        predictions_key_name=predictions_key_name,
        task_num=task_num,
    )
