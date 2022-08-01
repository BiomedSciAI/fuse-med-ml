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
from typing import List, Tuple, Union, Optional
from fuse.data.ops.op_base import OpBase, OpReversibleBase, op_call, op_reverse
from fuse.utils.misc.context import DummyContext
from fuse.utils.ndict import NDict
from fuse.utils.cpu_profiling.timer import Timer


class PipelineDefault(OpReversibleBase):
    """
    Pipeline default implementation
    Pipeline to run sequence of ops with a dictionary passing information between the ops.
    See OpBase for more information
    """

    def __init__(
        self,
        name: str,
        ops_and_kwargs: List[Tuple[OpBase, dict]],
        op_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """
        :param name: pipeline name
        :param ops_and_args: List of tuples. Each tuple include op and dictionary includes op specific arguments.
        :param op_ids: Optional, set op_id - unique name for every op. If not set, an index will be used
        :param verbose: set to True for debug messages such as the running time of each operation
        """
        super().__init__()
        self._name = name
        self._ops_and_kwargs = ops_and_kwargs
        if op_ids is None:
            self._op_ids = [str(index) for index in range(len(self._ops_and_kwargs))]
        else:
            assert len(self._ops_and_kwargs) == len(op_ids), "Expecting op_id for every op"
            assert len(set(op_ids)) == len(op_ids), "Expecting unique op id for every op."
            self._op_ids = op_ids
        self._verbose = verbose

    def extend(self, ops_and_kwargs: List[Tuple[OpBase, dict]], op_ids: Optional[List[str]] = None):
        """
        Extends pipeline
        :param ops_and_args: Ops to append, List of tuples. Each tuple include op and dictionary includes op specific arguments.
        :param op_ids: Optional, set op_id - unique name for every op. If not set, an index will be used
        """
        if op_ids is None:
            op_ids = [str(index + len(self._ops_and_kwargs)) for index in range(len(ops_and_kwargs))]
        else:
            assert len(ops_and_kwargs) == len(op_ids), "Expecting op_id for every op"
            all_op_ids = self._op_ids + op_ids
            assert len(set(all_op_ids)) == len(all_op_ids), "Expecting unique op id for every op."

        self._ops_and_kwargs.extend(ops_and_kwargs)
        self._op_ids.extend(op_ids)

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        text = []
        for (op_id, op_kwargs) in zip(self._op_ids, self._ops_and_kwargs):
            op, kwargs = op_kwargs
            text.append(str(op_id) + "@" + op.get_hashable_string_representation() + "@" + str(kwargs) + "@")
        return "".join(text)  # this is faster than accumulate_str+=new_str

    def __call__(
        self, sample_dict: NDict, op_id: Optional[str] = None, until_op_id: Optional[str] = None
    ) -> Union[None, dict, List[dict]]:
        """
        See super class
        plus
        :param until_op_id: optional - stop after the specified op_id - might be used for optimization
        """
        # set op_id if not specified
        if op_id is None:
            op_id = f"internal.{self._name}"

        samples_to_process = [sample_dict]
        for sub_op_id, (op, op_kwargs) in zip(self._op_ids, self._ops_and_kwargs):
            if self._verbose:
                context = Timer(f"Pipeline {self._name}: op {type(op).__name__}, op_id {sub_op_id}", self._verbose)
            else:
                context = DummyContext()
            with context:
                samples_to_process_next = []

                for sample in samples_to_process:

                    sample = op_call(op, sample, f"{op_id}.{sub_op_id}", **op_kwargs)

                    # three options for return value:
                    # None - ignore the sample
                    # List of dicts - split sample
                    # dict - modified sample
                    if sample is None:
                        return None
                    elif isinstance(sample, list):
                        samples_to_process_next += sample
                    elif isinstance(sample, dict):
                        samples_to_process_next.append(sample)
                    else:
                        raise Exception(f"unexpected sample type returned by {type(op)}: {type(sample)}")

            # continue to process with next op
            samples_to_process = samples_to_process_next

            # if required - stop after the specified op id
            if until_op_id is not None and sub_op_id == until_op_id:
                break

        # if single sample - return it, otherwise return list of samples.
        if len(samples_to_process) == 1:
            return samples_to_process[0]
        else:
            return samples_to_process

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str] = None) -> dict:
        """
        See super class
        """
        # set op_id if not specified
        if op_id is None:
            op_id = f"internal.{self._name}"

        for sub_op_id, (op, _) in zip(reversed(self._op_ids), reversed(self._ops_and_kwargs)):
            sample_dict = op_reverse(op, sample_dict, f"{op_id}.{sub_op_id}", key_to_reverse, key_to_follow)

        return sample_dict
