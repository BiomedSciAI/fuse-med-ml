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

from fuse.data.datasets.dataset_base import DatasetBase

from fuse.utils.file_io.file_io import save_dataframe


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
        **dataset_get_kwargs
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
        data = dataset.get_multi(keys=all_keys, desc="export", **dataset_get_kwargs)

        # store in dataframe
        df = pds.DataFrame()

        for key in all_keys:
            values = [sample_dict[key] for sample_dict in data]
            df[key] = values

        if output_filename is not None:
            save_dataframe(df, output_filename)

        return df
