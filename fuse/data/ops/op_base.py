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
from fuse.data.ops import get_function_call_str
from inspect import stack
from fuse.data.ops.caching_tools import get_callers_string_description, value_to_string
from fuse.utils.ndict import NDict

class OpBase(ABC):
    """
    Operator Base Class
    Operators are the building blocks of the sample processing pipeline.
    Each operator gets as an input the sample_dict as created be the previous operators
    and can either add/delete/modify fields in sample_dict.
    """

    _MISSING_SUPER_INIT_ERR_MSG = 'Did you forget to call super().__init__() ? Also, make sure you call it BEFORE setting any attribute.'
    
    def __init__(self, value_to_string_func: Callable = value_to_string):
        '''
        :param value_to_string_func: when init is called, a string representation of the caller(s) init args are recorded.
        This is used in __str__ which is used later for hashing in caching related tools (for example, SamplesCacher)
        value_to_string_func allows to provide a custom function that converts a value to string.
        This is useful if, for example, a custom behavior is desired for an object like numpy array or DataFrame.
        The expected signature is: foo(val:Any) -> str
        '''
        
        #the following is used to extract callers args, for __init__ calls up the stack of classes inheirting from OpBase
        #this way it can happen in the base class and then anyone creating new Ops will typically only need to add 
        #super().__init__ in their __init__ implementation
        self._stored_init_str_representation = get_callers_string_description(
            max_look_up=4,
            expected_class=OpBase,
            expected_function_name='__init__',
            value_to_string_func = value_to_string_func
            )
        
    def __setattr__(self, name, value):
        '''
        Verifies that super().__init__() is called before setting any attribute
        '''
        storage_name = '_stored_init_str_representation'
        if name != storage_name and not hasattr(self, storage_name):
            raise Exception(OpBase._MISSING_SUPER_INIT_ERR_MSG)
        super().__setattr__(name, value)

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

    def __str__(self) -> str:
        '''
        A string representation of this operation, which will be used for hashing.
        It includes recorded (string) data describing the args that were used in __init__()
        you can override/extend it in the rare cases that it's needed

        example:

        class OpSomethingNew(OpBase):
            def __init__(self):
                super().__init__()
            def __str__(self):
                ans = super().__str__(self)
                ans += 'whatever you want to add"

        '''

        if not hasattr(self, '_stored_init_str_representation'):
            raise Exception(OpBase._MISSING_SUPER_INIT_ERR_MSG)       
        call_repr = get_function_call_str(self.__call__, )

        return f'init_{self._stored_init_str_representation}@call_{call_repr}'

    
