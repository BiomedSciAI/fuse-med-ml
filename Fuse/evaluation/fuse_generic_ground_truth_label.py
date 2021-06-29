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

from typing import List

import pandas as pd

try:
    from class_definitions import class_definition
    from class_definitions.label import Label

    CANNOT_LOAD_CLASSEVE = False
except Exception:
    CANNOT_LOAD_CLASSEVE = True


class FuseGenericGroundTruthLabel(Label):
    """
    Generic class for checking that the value in a given GT column is contained in the allowed values list
    """

    def __init__(self, gt_column: str, allowed_values: List) -> None:
        """
        Init the column that contains the ground truth data and its values

        :param gt_column: the column with ground truth data
        :param allowed_values: the values that apply to this ground truth
        """
        if CANNOT_LOAD_CLASSEVE:
            raise Exception("ClassEve Cannot be loaded, please make sure it's on the PYTHONPATH")
        self.gt_column = gt_column
        self.allowed_values = allowed_values
        pass

    def __repr__(self):
        return f'{self.gt_column}-{self.allowed_values}'

    def match_entity(self, raw_data: pd.DataFrame) -> pd.Series:
        """
        Check that values in self.gt_column are in the self.allowed_values list
        :param raw_data: data to match
        :return: True where the raw_data gt_column is within the allowed values
        """
        return raw_data[self.gt_column].isin(self.allowed_values)


def get_generic_binary_class_definition(title: str = 'Positive VS Negative GT') -> class_definition.BinaryClassDefinition:
    """
    Create a generic class definition based on ground_truth column.
    positive label is ground_truth=1 and negative label is ground_truth=0

    :param title: title of the task
    :return: the binary class definition
    """
    # create the class_definition instance
    positive_label = FuseGenericGroundTruthLabel('ground_truth', [1])
    negative_label = FuseGenericGroundTruthLabel('ground_truth', [0])
    the_class_definition = class_definition.BinaryClassDefinition(title, positive_label=positive_label, negative_label=negative_label)
    return the_class_definition


def get_generic_multi_class_definition(label_names: List) -> class_definition.MultiClassDefinition:
    """
    Create a generic class definition based on ground_truth column.
    the created class definition will have len(label_names) classes, where the colum ground_truth is compared to class_name idx.
    For instance, given label_names = ['cyst', 'hemangioma', 'rest']
    the labels that will be created are [
        FuseGenericGroundTruthLabel('ground_truth', [0]),
        FuseGenericGroundTruthLabel('ground_truth', [1]),
        FuseGenericGroundTruthLabel('ground_truth', [2])]

    :param label_names: list of label names (for generating the labels and the title)
    :return: the multi class definition
    """
    # create the class_definition instance
    labels = [FuseGenericGroundTruthLabel('ground_truth', [idx]) for idx in range(len(label_names))]
    title = ' '.join([label_name for label_name in label_names])
    the_class_definition = class_definition.MultiClassDefinition(title=title, labels=labels)
    return the_class_definition
