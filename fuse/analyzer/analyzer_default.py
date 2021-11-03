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
import traceback
from typing import Callable, Dict, Hashable, Optional, Sequence, Union

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import trange

from fuse.data.data_source.data_source_from_list import FuseDataSourceFromList
from fuse.data.dataset.dataset_base import FuseDatasetBase
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.processor.processor_base import FuseProcessorBase
from fuse.data.processor.processor_dataframe import FuseProcessorDataFrame
from fuse.metrics.metric_base import FuseMetricBase
from fuse.utils.utils_debug import FuseUtilsDebug
from fuse.utils.utils_file import FuseUtilsFile
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict


class FuseAnalyzerDefault:
    """
    Default implementation of generic analyzer
    """

    def __init__(self):
        pass

    def analyze_prediction_dataframe(self, sample_descr: Union[Sequence[Hashable], str], dataframes_dict: Dict[str, pd.DataFrame], post_processing: Callable, 
                                     metrics: Dict[str, FuseMetricBase], output_filename: Optional[str] = None) -> Dict:
        """
        analyze predictions specified in a dataframe.
        Typically, but not a must, two dataframes will be provided in dataframes - one including the predictions one the targets
        Each sample will be composed from the values specified in all dataframes_dict, the key will for each values will be <dataframe key>.<column in dataframe>
        Processing the sample values is possible using post_processing function that gets samples_dict and returns modified sample_dict.
        
        :param samples_descr_source: will run the evaluation on the specified list of sample descriptors. Supported values:
                                 * dataframe name -  to evaluate all the samples specified in the specified dataframe.
                                 * list of samples descriptors to define the samples explicitly
        :param dataframes_dict: pairs of dataframe name and dataframe. The sample_descr column should be marked as the index using 'df = df.set_index(<column name>, drop=False)'
        :param post_processing: a callable getting a dict with all the values of a sample to post process it before evaluation
        :param metrics: metrics to compute. pairs of metric name and a metric
        :param output_filename: specify a filename to save a pickled dataframe including the results.
        :return: dataframe including the results
        """
        processors = {name: FuseProcessorDataFrame(df, sample_desc_column=df.index.name) for name, df in dataframes_dict.items()}
        
        if not isinstance(sample_descr, str):
            data_source = FuseDataSourceFromList(sample_descr)
        else:
            sample_ids = processors[sample_descr].get_samples_descriptors()
            data_source = FuseDataSourceFromList(sample_ids)

        # verify all samples are represented in all dataframes
        # create dataset
        dataset = FuseDatasetDefault(cache_dest=None,
                                     data_source=data_source,
                                     input_processors=None,
                                     gt_processors=None,
                                     processors=processors,
                                     post_processing_func=post_processing,
                                     data_key_prefix=None)
        dataset.create()

        return self.analyze_generic(dataset, None, metrics, output_filename, key_sample_desc="descriptor")



    def analyze(self, gt_processors: Dict[str, FuseProcessorBase], metrics: Dict[str, FuseMetricBase],
                data: Optional[pd.DataFrame] = None, data_pickle_filename: Optional[str] = None,
                output_filename: Optional[str] = None, **kwargs) -> Dict:
        """
        analyze all the samples included in the infer file.
        :param gt_processors: processors that generates the ground truth data required for analyzing
        :param metrics: dictionary of required metrics
        :param data:  input DataFrame
        :param data_pickle_filename: path to a pickled DataFrame (possible gzipped)
        :param output_filename: path used to save the report
        :param kwargs: additional parameters specified in function analyze_generic()
        :return: collected results
        """
        # create inference processor
        infer_processor = FuseProcessorDataFrame(data=data, data_pickle_filename=data_pickle_filename)

        # create data source
        descriptors_list = infer_processor.get_samples_descriptors()
        data_source = FuseDataSourceFromList(descriptors_list)

        # create dataset
        dataset = FuseDatasetDefault(cache_dest=None,
                                     data_source=data_source,
                                     input_processors={},
                                     gt_processors=gt_processors)
        dataset.create()

        # run analyze generic
        return self.analyze_generic(dataset, infer_processor, metrics, output_filename, **kwargs)

    def analyze_generic(self, dataset: FuseDatasetBase, infer_processor: FuseProcessorBase, metrics: Dict[str, FuseMetricBase],
                        output_filename: Optional[str] = None, output_file_mode: str = 'w', print_results: bool = True,
                        num_workers: int = 4, batch_size: int = 2, key_sample_desc: str="data.descriptor", ) -> Dict:
        """
        analyze all the samples in the dataset.
        :param dataset: dataset object - does not need to include the 'input' data.
                        For example DataSetDefault instance can be created only with the ground truth processors
        :param infer_processor: processor extracting the data generated by manager.infer()
                                Typically will be instance of FuseProcessorCSV
        :param metrics: dictionary of required metrics
        :param output_filename: path used to save the report
        :param output_file_mode: either write mode ('w') or append model ('a')
        :param num_workers: number of dataloader workers
        :param batch_size: batch_size - data is loaded and collected in batches
        :param print_results: print metrics results to screen
        :return: collected results
        """
        # debug - num workers
        override_num_workers = FuseUtilsDebug().get_setting('manager_override_num_dataloader_workers')
        if override_num_workers != 'default':
            num_workers = override_num_workers
            logging.getLogger('Fuse').info(f'Manager - debug mode - override dataloader num_workers to {override_num_workers}', {'color': 'red'})

        # create dataloader
        dataloader = DataLoader(dataset=dataset,
                                shuffle=False, drop_last=False,
                                batch_size=batch_size,
                                collate_fn=dataset.collate_fn,
                                num_workers=num_workers)

        # reset metrics
        for metric in metrics.values():
            metric.reset()

        # iterate and collect batches
        data_iter = iter(dataloader)
        for _ in trange(len(dataloader)):
            batch_dict = next(data_iter)
            self.analyze_batch(batch_dict, infer_processor, metrics, key_sample_desc)

        # metric process
        metrics_results = {metric_name: metric.process() for metric_name, metric in metrics.items()}

        # print results
        if print_results:
            self.print_metrics_results(metrics_results)

        lgr = logging.getLogger('Fuse')

        # dump metrics results
        if output_filename is not None:
            self.dump_metrics_results(metrics_results, output_filename, output_file_mode)
            lgr.info(f'\nAnalyzer done. Results saved in {output_filename}  ', {'color': 'magenta', 'attrs': 'bold'})

        lgr.info(f'\nAnalyzer done.', {'color': 'magenta', 'attrs': 'bold'})
        return metrics_results

    def analyze_batch(self, batch_dict: dict, infer_processor: FuseProcessorBase,
                      metrics: Dict[str, FuseMetricBase], key_sample_desc: str) -> None:
        """
        Static function analyzing batch of samples
        For details about the input parameters can be found in FuseAnalyzerDefault.analyze_generic()
        """
        try:
            # get the sample descriptor of the sample
            sample_descriptors = FuseUtilsHierarchicalDict.get(batch_dict, key_sample_desc)

            if infer_processor is not None:
                # get the infer data
                infer_batch_data = [infer_processor(descr) for descr in sample_descriptors]
                # add infer data to batch_dict
                for key in infer_batch_data[0]:
                    FuseUtilsHierarchicalDict.set(batch_dict, key, [infer_sample_data[key] for infer_sample_data in infer_batch_data])

            # metric collect operation
            for metric in metrics.values():
                metric.collect(batch_dict)
        except:
            lgr = logging.getLogger('Fuse')
            # do not stop analyzing - print the error and continue
            track = traceback.format_exc()
            lgr.error(track)
            lgr.error(f'Failed to analyze batch, skipping')

    def print_metrics_results(self, metrics_results: Dict) -> None:
        """
        Print the metrics results
        :param metrics_results: results return from metric.process()
        :return: None
        """
        lgr = logging.getLogger('Fuse')
        lgr.info(f'Results', {'attrs': ['bold', 'underline']})
        for metric_name, metric_result in metrics_results.items():
            lgr.info(f'\nMetric {metric_name}:', {'attrs': 'bold'})
            if isinstance(metric_result, dict):
                keys = FuseUtilsHierarchicalDict.get_all_keys(metric_result)
                for key in keys:
                    lgr.info(f'{key}: {FuseUtilsHierarchicalDict.get(metric_result, key)}')
            else:
                lgr.info(metric_result)

    def dump_metrics_results(self, metrics_results: Dict, filename: str, output_file_mode: str = 'w') -> None:
        """
        Dump results to a file
        :param metrics_results: results return from metric.process()
        :param filename: output file name
        :param output_file_mode: either write mode ('w') or append model ('a')
        :return: None
        """
        # make sure the output folder exist
        dir_path = os.path.dirname(filename)
        FuseUtilsFile.create_dir(dir_path)

        with open(filename, output_file_mode) as output_file:
            output_file.write(f'Results:\n')

            for metric_name, metric_result in metrics_results.items():
                output_file.write(f'\nMetric {metric_name}:\n')
                output_file.write(f'------------------------------------------------\n')
                if isinstance(metric_result, dict):
                    keys = FuseUtilsHierarchicalDict.get_all_keys(metric_result)
                    for key in keys:
                        output_file.write(f'{key}: {FuseUtilsHierarchicalDict.get(metric_result, key)}\n')
                else:
                    output_file.write(f'{metric_result}\n')
