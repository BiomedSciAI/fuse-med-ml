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

"""
Augmentor Default class
"""
from typing import Any, Iterable

from fuse.data.augmentor.augmentor_base import FuseAugmentorBase
from fuse.utils.utils_hierarchical_dict import FuseUtilsHierarchicalDict
from fuse.utils.utils_logger import log_object_input_state, convert_state_to_str
from fuse.utils.utils_param_sampler import sample_all


class FuseAugmentorDefault(FuseAugmentorBase):
    """
    Default generic implementation for Fuse augmentor. Aimed to be used by most experiments.
    """

    def __init__(self, augmentation_pipeline: Iterable[Any] = ()):
        """
        :param augmentation_pipeline: list of augmentation operation description,
        Each operation description expected to be a tuple of 4 elements:
            Element 0 - the sample keys affected by this operation
            Element 1 - callback to a function performing the operation. This function expected to support input parameter 'aug_ingput'
            Element 2 - dictionary including the input parameters for the callback function. See AugmentorSamplerDefault
                        to learn how to use random numbers
            Element 3 - general parameters: TBD

            Example:
                See in aug_image_default_pipeline()
        """
        # log object input state
        log_object_input_state(self, locals())

        self.augmentation_pipeline = augmentation_pipeline

    def get_random_augmentation_desc(self) -> Any:
        """
        See description in super class.
        """
        return sample_all(self.augmentation_pipeline)

    def apply_augmentation(self, sample: Any, augmentation_desc: Any) -> Any:
        """
        See description in super class.
        """
        aug_sample = sample
        for op_desc in augmentation_desc:
            # decode augmentation description
            sample_keys = op_desc[0]
            augment_function = op_desc[1]
            augment_function_parameters = op_desc[2]
            general_parameters: dict = op_desc[3]

            # If apply sampled as False skip - by default it will always be True
            apply = general_parameters.get('apply', True)
            if not apply:
                continue

            # Extract augmentation input
            if sample_keys is None:
                aug_input = aug_sample
            elif len(sample_keys) == 1:
                aug_input = FuseUtilsHierarchicalDict.get(aug_sample, sample_keys[0])
            else:
                aug_input = tuple((FuseUtilsHierarchicalDict.get(aug_sample, key) for key in sample_keys))
            augment_function_parameters = augment_function_parameters.copy()
            augment_function_parameters['aug_input'] = aug_input

            # apply augmentation
            aug_result = augment_function(**augment_function_parameters)

            # modify the sample accordingly
            if sample_keys is None:
                aug_sample = aug_result
            elif len(sample_keys) == 1:
                FuseUtilsHierarchicalDict.set(aug_sample, sample_keys[0], aug_result)
            else:
                for index, key in enumerate(sample_keys):
                    FuseUtilsHierarchicalDict.set(aug_sample, key, aug_result[index])

        return aug_sample

    def summary(self) -> str:
        """
        String summary of the object
        """
        return \
            f'Class = {self, __class__}\n' \
                f'Pipeline = {convert_state_to_str(self.augmentation_pipeline)}'
