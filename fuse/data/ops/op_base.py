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
from typing import Dict, Union, List, Sequence, Any, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
from collections import OrderedDict
from fuse.data.patterns import Patterns
from fuse.utils.ndict import NDict
from fuse.data.ops.hashable_class import HashableClass

class OpBase(HashableClass):
    """
    Operator Base Class
    Operators are the building blocks of the sample processing pipeline.
    Each operator gets as an input the sample_dict as created be the previous operators
    and can either add/delete/modify fields in sample_dict.
    """    

    @abstractmethod
    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        call function that apply the operation
        :param sample_dict: the generated dictionary generated so far (generated be the previous ops in the pipeline)
                            The first op will typically get just the sample_id stored in sample_dict['data']['sample_id']
        :param op_id: unique identifier for an operation.
        Might be used to support reverse operation as sample_dict key in case information should be stored in sample_dict.
        In such a case use sample_dict[op_id] = info_to_store
        :param kwargs: additional arguments defined per operation
        :return: Typically modified sample_dict.
                There are two special cases supported only if the operation is in static pipeline:
                * return None - ignore the sample and do not raise an error
                * return list of sample_dict - a case splitted to few samples. for example image splitted to patches.
        """
        raise NotImplementedError

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        reverse operation
        If a reverse operation is not necessary (for example operator that reads an image),
        just implement a reverse method that does nothing.

        If reverse operation is necessary but not required by the project,
        keep the base implementation which will throw an NotImplementedError in case the reverse operation will be called.

        To support reverse operation, store the parameters which necessary to apply the reverse operation
        such as key to the transformed value and the argument to the transform operation in sample_dict[op_id].
        Those values can be extracted back during the reverse operation.

        :param sample_dict: the dictionary as modified by the previous steps (reversed direction)
        :param op_id: See op_id in __call__ function
        :param key_to_reverse: the required value to reverse
        :param key_to_follow: run the reverse according to the operation applied on this value
        :return: modified sample_dict
        """
        raise NotImplemented

    

    
