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
from typing import Optional, Sequence
import pandas as pds
import torch
import os
from fuse.data.datasets.dataset_base import DatasetBase

from fuse.utils.file_io.file_io import save_dataframe, save_pickle_safe, load_pickle


class ExportDataset:
    """
    Export data
    """

    @staticmethod
    def export_to_dataframe(
        dataset: DatasetBase,
        keys: Sequence[str],
        output_filename: Optional[str] = None,
        sample_id_key: str = "data.sample_id",
        **dataset_get_kwargs,
    ) -> pds.DataFrame:
        """
        extract from dataset the specified and keys and create a dataframe.
        If output_filename will be specified, the dataframe will also be saved in a file.
        :param dataset: the dataset to extract the values from
        :param keys: keys to extract from sample_dict
        :param output_filename: Optional, if set, will save the dataframe into a file.
                                The file type will be inferred from filename, see fuse.utils.file_io.file_io.save_dataframe for more details
        :param dataset_get_kwargs: additional parameters to dataset.get(), might be used to optimize the running time
        """
        # add sample_id to keys list
        if keys is not None:
            all_keys = []
            all_keys += list(keys)
            if sample_id_key not in keys:
                all_keys.append(sample_id_key)
        else:
            all_keys = None

        # read all the data
        data = dataset.get_multi(keys=all_keys, **dataset_get_kwargs)

        # store in dataframe
        df = pds.DataFrame()

        for key in all_keys:
            values = [sample_dict[key] for sample_dict in data]
            df[key] = values

        if output_filename is not None:
            save_dataframe(df, output_filename)

        return df

    @staticmethod
    def export_to_dir(
        dataset: DatasetBase,
        output_dir: str,
        keys: Optional[Sequence[str]] = None,
        sample_id_key: str = "data.sample_id",
        **dataset_get_kwargs,
    ):
        """
        extract from dataset the specified and keys and writes to a specified directory in the disk
        :param dataset: the dataset to extract the values from
        :param keys: Optional, keys to extract from sample_dict. If None - all keys will be saved
        :param output_dir: Optional, if set, will save the dataframe into a file.
        :param dataset_get_kwargs: additional parameters to dataset.get(), might be used to optimize the running time
        """
        # add sample_id to keys list
        if keys is not None:
            all_keys = []
            all_keys += list(keys)
            if sample_id_key not in keys:
                all_keys.append(sample_id_key)
        else:
            all_keys = None

        # read all the data
        data = dataset.get_multi(keys=all_keys, **dataset_get_kwargs)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for i, sample_dict in enumerate(data):
            sample_id = sample_dict[sample_id_key]
            sample_dict = sample_dict.flatten()

            if all_keys is not None:
                sample_dict = {sample_dict[k] for k in all_keys}

            d2 = {}
            for k, v in sample_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                d2[k] = v
            output_file = os.path.join(output_dir, f"{sample_id}.pkl")
            save_pickle_safe(d2, output_file)
            print(i, "wrote", output_file)
        print("done")
