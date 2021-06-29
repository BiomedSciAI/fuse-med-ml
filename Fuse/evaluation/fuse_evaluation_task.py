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

from typing import Callable

from Fuse.data.processor.processor_base import FuseProcessorBase
from Fuse.evaluation.fuse_generic_ground_truth_label import get_generic_binary_class_definition
from Fuse.evaluation.fuse_prediction_results import FusePredictionResults

try:
    from class_definitions import class_definition
    from data_entities.evaluation_task import EvaluationTask

    CANNOT_LOAD_CLASSEVE = False
except Exception:
    CANNOT_LOAD_CLASSEVE = True


class FuseEvaluationTask(EvaluationTask):
    """
    Basic Binary Class Evaluation Task Definition.
    """

    def __init__(self, prediction_results: FusePredictionResults = None, class_def: class_definition.MultiClassDefinition = None,
                 inference_filename: str = None, output_column: str = None,
                 gt_processor: FuseProcessorBase = None, weights_function: Callable = None, task_title: str = None):
        """
        Creates an instance of Evaluation Task.
        For this, it creates a FusePredictionResults and a BinaryClassDefinition using get_generic_binary_class_definition()

        :param prediction_results: unified prediction results.
            When None, the prediction results object is initialized from inference_filename, output_column, gt_processor, weights_function
        :param class_def: class definition for the evaluation task.
            When None, a generic binary class definition is created with the title task_title.
        :param inference_filename: prediction results input file
        :param output_column: column to consider as the score to be analyzed
        :param gt_processor: processor of ground truth
        :param weights_function: function to compute sample weights
        :param task_title: title for binary class definition
        """
        if CANNOT_LOAD_CLASSEVE:
            raise Exception("ClassEve Cannot be loaded, please make sure it's on the PYTHONPATH")

        if prediction_results is not None:
            self.prediction_results = prediction_results
        else:
            self.prediction_results = FusePredictionResults(inference_filename=inference_filename, output_column=output_column,
                                                            gt_processor=gt_processor, weights_function=weights_function)
        if class_def is not None:
            self.class_definition = class_def
        else:
            self.class_definition = get_generic_binary_class_definition(task_title)

        super().__init__(self.prediction_results, self.class_definition, weights_function, pos_label_idx=1)
        pass

    def get_unified_prediction(self) -> FusePredictionResults:
        """
        returns the unified prediction resutls that this task is based on
        :return: unified prediction result
        """
        return self.prediction_results

    def get_class_definition(self) -> class_definition.BinaryClassDefinition:
        """
        Returns the binary class definition that this task is based on
        :return: class definition
        """
        return self.class_definition
