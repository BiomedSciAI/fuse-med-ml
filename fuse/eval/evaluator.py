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

import os
from typing import Dict, Hashable, Iterable, List, Optional, OrderedDict, Sequence, Union
import pickle
import pandas as pd

from fuse.utils import read_dataframe
from fuse.utils import NDict
from fuse.utils import CollateToBatchList

from fuse.eval.metrics.metrics_common import MetricBase


class EvaluatorDefault:
    """
    Evaluator
    Combine all the input sources, evaluate using the specified metrics,
    generate a report and returns a dictionary with all the metrics results.
    """

    def __init__(self) -> None:
        pass

    def eval(
        self,
        ids: List[Hashable],
        data: Union[
            str,
            pd.DataFrame,
            Sequence[str],
            Sequence[pd.DataFrame],
            Iterable,
            Dict[str, Union[str, pd.DataFrame, Sequence[str], Sequence[pd.DataFrame]]],
        ],
        metrics: OrderedDict[str, MetricBase],
        id_key: str = "id",
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        silent: bool = True,
    ) -> NDict:
        """
        evaluate, return, print and optionally dump results to a file
        :param ids: list if sample_ids to consider
        :param data: data to evaluate
                     Supported formats:
                        (1) str - path to a file -  csv, pickled dataframe or json. Assumes column called 'id' that holds unique identifier for a sample
                        (2) dataframe - assumes column called 'id'  that holds unique identifier for a sample
                        (3) Sequence of (1) or (2) - each element in the sequence considered to be a different fold.
                                                    ("evaluator_fold" will be added automatically to store the fold number)
                        (4) Dict of (1), (2)  or (3) - each item (key-value) in the dict may contain different information about the same sample.
                                                   A prefix of "<key>." will be automatically added to each column name to avoid from collisions.
                        (5) Iterator - either sample iterator or batch iterator. See param batch_size documentation for more details.

        :param id_key: the column name or key that include the sample id (unique identifier)
        :param batch_size: Optional. Use when the data is too big to read together,
                           Options:
                                None - read all the data together
                                0 - read each batch separately.
                                    data must be an iterator of batch_dict. batch_dict is a dictionary (or nested dictionary).
                                    Each item in this dictionary must include value per sample in the batch.
                                integer > 0 - read each sample separately.
                                              data must be an iterator of samples.
                                              A batch will be automatically created from batch_size samples
        :param output_dir: Optional - dump results to directory
        :param silent: print results if false
        :return: dictionary that holds all the results.
        """
        self.silent = silent

        if ids is not None:
            ids_df = pd.DataFrame(ids, columns=[id_key]).set_index(id_key)  # use the specified samples
        else:
            ids_df = None  # use all samples

        if batch_size is None:
            data_df = self.read_data(data, ids_df, id_key=id_key)
            data_df["id"] = data_df[id_key]

            # pass data
            for metric_name, metric in metrics.items():
                try:
                    metric.set(data_df)
                except:
                    print(f"Error: metric {metric_name} set() method failed")
                    raise
        elif batch_size > 0:
            # data is iterable - each iteration is a single sample (represented by a dictionary)
            assert isinstance(data, Iterable), "Error: batch mode (batch_size != None) supports only iterable data"
            collate_fn = CollateToBatchList()
            data_iter = iter(data)
            while True:
                samples = []
                try:
                    for _ in range(batch_size):
                        sample = next(data_iter)
                        sample["id"] = sample[id_key]
                        samples.append(sample)
                except StopIteration:
                    break
                finally:
                    if len(samples) > 0:
                        batch = collate_fn(samples)
                        for metric in metrics.values():
                            metric.collect(batch)
        else:
            # data is iterable - each iteration is batch dict (each used element in dict should have batch dimension or list of size batch size)
            assert isinstance(data, Iterable), "Error: batch mode (batch_size != None) supports only iterable data"
            data_iter = iter(data)
            while True:
                try:
                    batch = next(data_iter)
                    for metric in metrics.values():
                        metric.collect(batch)
                except StopIteration:
                    break

        # evaluate
        results = NDict()
        for metric_name, metric in metrics.items():
            try:
                results[f"metrics.{metric_name}"] = metric.eval(results, ids)
            except:
                print(f"Error: metric {metric_name} eval() method failed")
                raise

        # dump results
        self.dump_metrics_results(results, output_dir)

        return results

    def read_data(
        self,
        data: Union[
            str,
            pd.DataFrame,
            Sequence[str],
            Sequence[pd.DataFrame],
            Dict[str, Union[str, pd.DataFrame, Sequence[str], Sequence[pd.DataFrame], Iterable]],
        ],
        ids_df: pd.DataFrame,
        id_key: str,
        error_missing_ids: bool = True,
        error_duplicate: bool = True,
    ):
        """
        Read data and convert to a single dataframe
        :param data: the input data to covert - see eval() method to understand the supported options
        :param ids_df: the returned dataframe will include only the ids specified in ids_df
        :param error_missing_ids: raise an error if data does not include some of the ids specified in ids_df
        :param error_duplicate: raise an error if id exist more than once in data
        """
        if isinstance(data, pd.DataFrame):  # data is already a dataframe
            result_data = data
            # make sure "id" column exist and set it as index
            if id_key not in result_data.keys():
                raise Exception(
                    "Error: 'id' column/key wasn't found in data. Index column specify unique identifier per sample"
                )
            result_data = result_data.set_index(keys=id_key, drop=False)

        elif isinstance(data, str):  # data is path to a file

            result_data = read_dataframe(data)
            # make sure "id" column exist and set it as index
            if id_key not in result_data.keys():
                raise Exception(
                    "Error: 'id' column/key wasn't found in data. Index column specify unique identifier per sample"
                )
            result_data = result_data.set_index(keys=id_key, drop=False)

        elif isinstance(data, Sequence) and isinstance(data[0], (str, pd.DataFrame)):  # data is a sequence of folds
            data_lst = []
            for fold, data_elem in enumerate(data):
                data_elem_df = self.read_data(data_elem, ids_df, error_missing_ids=False, id_key=id_key)
                # add "fold" number to dataframe
                data_elem_df["evaluator_fold"] = fold
                data_lst.append(data_elem_df)
            result_data = pd.concat(data_lst)

        elif isinstance(data, dict):  # data is dictionary of dataframes
            df_list = []
            all_ids = set()
            for key, data_elem in data.items():

                try:
                    data_elem_df = self.read_data(data_elem, ids_df, id_key=id_key)
                    all_ids = all_ids.union(set(data_elem_df.index))
                except:
                    print(f"Error: key={key}")
                    raise

                data_elem_df = data_elem_df.add_prefix(key + ".")
                df_list.append(data_elem_df)

            # make sure ids exists in all dataframes
            for data_elem_df in df_list:
                missing_ids = all_ids - set(data_elem_df.index)
                if len(missing_ids) > 0:
                    raise Exception(
                        f"Error: ids {missing_ids} are missing in data['{data_elem_df.keys()[0].split('.')[0]}']"
                    )

            result_data = pd.concat(df_list, axis=1, join="inner")
            result_data[id_key] = result_data.index

        elif isinstance(data, Iterable):
            result_data = pd.DataFrame([dict(elem) for elem in data])

            # make sure "id" column exist and set it as index
            if id_key not in result_data.keys():
                raise Exception(
                    f"Error: {id_key} column/key wasn't found in data. Index column specify unique identifier per sample - available keys are {result_data.keys()}"
                )
            result_data = result_data.set_index(keys=id_key, drop=False)

        else:
            raise Exception(f"Error: unexpected data type {type(data)}")

        if ids_df is not None:
            # filter to include only the required ids
            result_data = result_data.loc[result_data.index.intersection(ids_df.index)]

            # make sure all ids exists
            if error_missing_ids and len(ids_df) > len(result_data):
                missing_ids = set(ids_df.index) - set(result_data.index)
                raise Exception(f"Error: missing samples in evaluation (missing_ids={missing_ids}")

        # raise error if the same id exist more than once
        if error_duplicate:
            duplicated = result_data.index.duplicated(keep=False)
            duplications = result_data.loc[duplicated].index.unique()
            if not duplications.empty:
                raise Exception(f"Error: found duplicated ids in data: {list(duplications)}")

        return result_data

    def dump_metrics_results(self, metrics_results: NDict, output_dir: str) -> None:
        """
        Dump results to a file
        :param metrics_results: results return from metric.process()
        :param output_dir: directory to dump files to
        :return: None
        """
        # metrics text results
        results = "Results:\n"

        for metric_name in metrics_results["metrics"]:
            # skip metrics use as intermidate computation
            if metric_name.startswith("_"):
                continue
            metric_result = metrics_results["metrics"][metric_name]
            results += f"\nMetric {metric_name}:\n"
            results += "------------------------------------------------\n"

            if isinstance(metric_result, dict):
                metric_result = NDict(metric_result)
                keys = metric_result.keypaths()
                for key in keys:
                    results += f"{key}:\n{metric_result[key]}\n"
            else:
                results += f"{metric_result}\n"

        # print to screen
        if not self.silent:
            print(results)

        # make sure the output folder exist
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            # save text results
            with open(os.path.join(output_dir, "results.txt"), "w") as output_file:
                output_file.write(results)

            # save pickled results
            with open(os.path.join(output_dir, "results.pickle"), "wb") as output_file:
                pickle.dump(metrics_results, output_file)
