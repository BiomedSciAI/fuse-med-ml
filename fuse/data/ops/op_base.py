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
from typing import Union, List, Optional
from abc import abstractmethod
from fuse.data.utils.sample import get_sample_id
from fuse.utils.ndict import NDict
from fuse.data.ops.hashable_class import HashableClass


class OpBase(HashableClass):
    """
    Operator Base Class
    Operators are the building blocks of the sample processing pipeline.
    Each operator gets as an input the sample_dict as created by the previous operator in pipeline,
    modify sample_dict (can either add/delete/modify fields in sample_dict)before passing it to the next operator in pipeline.
    """

    @abstractmethod
    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:
        """
        call function that apply the operation
        :param sample_dict: the generated dictionary generated so far (generated be the previous ops in the pipeline)
                            The first op will typically get just the sample_id stored in sample_dict['data']['sample_id']
        :param kwargs: additional arguments defined per operation
        :return: Typically modified sample_dict.
                There are two special cases supported only if the operation is in static pipeline:
                * return None - ignore the sample and do not raise an error
                * return list of sample_dict - a case splitted to few samples. for example image splitted to patches.
        """
        raise NotImplementedError


class OpReversibleBase(OpBase):
    """
    Special case of op - declaring that the operation can be reversed when required
    (useful to reverse processing steps before presenting the output)
    If there is nothing to reverse - to just declare that the op is reversible inherit from  OpReversibleBase instead of OpBase and implement simple reverse method that returns sample_dict as is.

    If some logic required to reverse the operation:
        (1) record the information required to reverse the operation in __call__ function. Use op_id to store it in sample_dict (sample_dict[op_id] = <information to record?).
        (2) override reverse() method: read the recorded information from sample_dict[op_id] and use it to reverse the operation
    """

    @abstractmethod
    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        See OpBase.__call__ for more infomation. The only difference is the extra argument that can be used to "record" the information required to reverse the operation.
        :param op_id: unique identifier for an operation.
                      Might be used to support reverse operation as sample_dict key.
                      In such a case use sample_dict[op_id] = info_to_store

        """
        raise NotImplementedError

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        reverse operation
        If a reverse operation is not necessary (for example operator that reads an image),
        implement simple reverse method that returns sample_dict as is

        If reverse operation is necessary but not required by the project so far,
        inherit from OpBase (will throw an NotImplementedError in case the reverse operation will be called).

        To support reverse operation, store the parameters which necessary to apply the reverse operation
        such as key to the transformed value and the argument to the transform operation in sample_dict[op_id].
        Those values can be extracted back during the reverse operation.

        :param sample_dict: the dictionary as modified by the previous steps (reversed direction)
        :param op_id: See op_id in __call__ function
        :param key_to_reverse: the required value to reverse
        :param key_to_follow: run the reverse according to the operation applied on this value
        :return: modified sample_dict
        """
        raise NotImplementedError(
            f"op {self} is not reversible. If there is nothing to reverse, just implement simple reverse method that returns sample_dict as is. If extra logic required to reverse follow the instructions in OpReversibleBase"
        )


def op_call(op: OpBase, sample_dict: NDict, op_id: str, **kwargs):
    try:
        if isinstance(op, OpReversibleBase):
            return op(sample_dict, op_id=op_id, **kwargs)
        else:  # OpBase but note reversible
            return op(sample_dict, **kwargs)
    except:
        # error messages are cryptic without this. For example, you can get "TypeError: __call__() got an unexpected keyword argument 'key_out_input'" , without any reference to the relevant op!
        print(
            "************************************************************************************************************************************\n"
            + f"error in __call__ method of op={op}, op_id={op_id}, sample_id={get_sample_id(sample_dict)} - more details below"
            + "*************************************************************************************************************************************\n"
        )
        raise


def op_reverse(op, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]):
    if isinstance(op, OpReversibleBase):
        try:
            return op.reverse(sample_dict, key_to_reverse, key_to_follow, op_id)
        except:
            # error messages are cryptic without this. For example, you can get "TypeError: __call__() got an unexpected keyword argument 'key_out_input'" , without any reference to the relevant op!
            print(
                f"error in reverse method of op={op}, op_id={op_id}, sample_id={get_sample_id(sample_dict)} - more details below"
            )
            raise

    else:  # OpBase but note reversible
        raise NotImplementedError(
            f"op {op} is not reversible. If there is nothing to reverse, just inherit OpReversibleBase instead of OpBase and implement simple reverse method that returns sample_dict as is. If extra logic required to reverse follow the instructions in OpReversibleBase"
        )
