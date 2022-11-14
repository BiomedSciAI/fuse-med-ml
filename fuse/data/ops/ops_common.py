import math
import numbers
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from fuse.data.key_types import TypeDetectorBase
import copy
from enum import Enum
from .op_base import OpBase, OpReversibleBase, op_call, op_reverse  # DataType,
from fuse.data.patterns import Patterns
from fuse.utils.ndict import NDict
import numpy as np


class OpRepeat(OpReversibleBase):
    """
    Repeat an op multiple times

    Typically used to apply the same operation on a list of keys in sample_dict
    Example:
    "

    repeat_for =

        #...
        (OpRepeat(OpCropToMinimalBBox(),
            [dict(key='data.cc.image'), dict(key='data.mlo.image'),dict(key='data.mlo.seg', margin=100)] #per provided dict a new OpCropToMinimalBBox invocation will be triggered
            )),
            dict(margin=12)), #this value will be passed to all OpCropToMinimalBBox invocations
        #...
    ]

    note - the values in provided in the list of dicts will *override* any kwargs
    In the example above, margin=12 will be used for both 'data.cc.image' and 'data.mlo.image',
        but a value of margin=100 will be used for 'data.mlo.seg'

    "
    """

    def __init__(self, op: OpBase, kwargs_per_step_to_add: Sequence[dict]):
        """
        See example above
        :param op: the operation to repeat
        :param kwargs_per_step_to_add: sequence of arguments (kwargs format) specific for a single repetition. those arguments will be added/overide the kwargs provided in __call__() function.
        """
        super().__init__()
        self._op = op
        self._kwargs_per_step_to_add = kwargs_per_step_to_add

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        See super class
        """

        for step_index, step_kwargs_to_add in enumerate(self._kwargs_per_step_to_add):
            step_kwargs = copy.copy(kwargs)
            step_kwargs.update(step_kwargs_to_add)
            full_step_id = f"{op_id}_{step_index}"
            sample_dict[full_step_id + "_debug_info.op_name"] = self._op.__class__.__name__
            sample_dict = op_call(self._op, sample_dict, full_step_id, **step_kwargs)

            assert not isinstance(
                sample_dict, list
            ), f"splitting samples within {type(self).__name__} operation is not supported"

            if sample_dict is None:
                return None
            elif not isinstance(sample_dict, dict):
                raise Exception(f"unexpected sample_dict type {type(sample_dict)}")

        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        See super class
        """
        for step_index in reversed(range(len(self._kwargs_per_step_to_add))):
            sample_dict = op_reverse(self._op, sample_dict, key_to_reverse, key_to_follow, f"{op_id}_{step_index}")

        return sample_dict


class OpLambda(OpReversibleBase):
    """
    Apply simple lambda function / function to transform single value from sample_dict (or the all dictionary)
    Optionally add reverse method if required.
    Example:
    OpLambda(func=lambda x: torch.tensor(x))
    """

    def __init__(self, func: Callable, func_reverse: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self._func = func
        self._func_reverse = func_reverse

    def __call__(
        self, sample_dict: NDict, op_id: Optional[str], key: Optional[str] = None, **kwargs
    ) -> Union[None, dict, List[dict]]:
        """
        More details in super class
        :param key: apply lambda func on sample_dict[key]. If none the input and output of the lambda function are the entire sample_dict
        """
        sample_dict[op_id] = key
        if key is not None:
            value = sample_dict[key]
            value = self._func(value, **kwargs)
            sample_dict[key] = value
        else:
            sample_dict = self._func(sample_dict)

        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        See super class
        """
        key = sample_dict[op_id]
        if key is not None:
            if key == key_to_follow:
                value = sample_dict[key_to_reverse]
                value = self._func_reverse(value)
                sample_dict[key_to_reverse] = value
        else:
            sample_dict = self._func_reverse(sample_dict)

        return sample_dict


class OpFunc(OpReversibleBase):
    """
    Helps to wrap an existing simple python function without writing boilerplate code.

    The wrapped function format is:

    def foo(*, *kwargs) -> Tuple:
        pass


    Example:

    def add_seperator(text:str, sep=' '):
        return sep.join(text)

    OpAddSeperator = OpFunc(add_seperator)

    usage in pipeline:

    pipeline = [
        (OpAddSeperator, dict(inputs={'data.text_input':'text'}, outputs='data.text_input'), #
    ]

    """

    def __init__(self, func: Callable, **kwargs):
        """
        :param func: a callable to call in  __call__()
        """
        super().__init__(**kwargs)
        self._func = func

    def __call__(
        self,
        sample_dict: NDict,
        op_id: Optional[str],
        inputs: Dict[str, str],
        outputs: Union[Sequence[str], str],
        **kwargs,
    ) -> Union[None, dict, List[dict]]:
        """
        See super class
        :param inputs: dictionary that map between the key_name of a value stored in sample_dict the the input argument name in func
        :param outputs: sequence of key_names to store each return value of func.

        """
        # extract inputs from sample dict
        kwargs_from_sample_dict = {}
        for input_key_name, func_arg_name in inputs.items():
            value = sample_dict[input_key_name]
            kwargs_from_sample_dict[func_arg_name] = value

        # all kwargs
        all_kwargs = copy.copy(kwargs)
        all_kwargs.update(kwargs_from_sample_dict)
        func_outputs = self._func(**all_kwargs)

        # add to sample_dict
        if isinstance(outputs, str):
            sample_dict[outputs] = func_outputs
        elif isinstance(outputs, Sequence):
            assert len(func_outputs) == len(
                outputs
            ), f"expecting that function {self._func} will output {len(outputs)} values"
            for output_name, output_value in zip(outputs, func_outputs):
                sample_dict[output_name] = output_value
        else:
            raise Exception(
                f"expecting outputs to be either str or sequence of str. got {type(self._outputs).__name__}"
            )

        return sample_dict


class OpApplyPatterns(OpReversibleBase):
    """
    Select and apply an operation according to key name.
    Instead of specifying every relevant key, the op will be applied for every key that matched a specified pattern
    Example:
    patterns_dict = OrderedDict([(r"^.*.cc.img$|^.*.cc.seg$", (op_affine, dict(rotate=Uniform(-90.0, 90.0))),
                                (r"^.*.mlo.img$|^.*.mlo.seg$", (op_affine, dict(rotate=Uniform(-45.0, 54.0)))])
    op_apply_pat = OpApplyPatterns(patterns_dict)
    """

    def __init__(self, patterns_dict: Optional[OrderedDict] = None):
        """
        :param patterns_dict: map a regex pattern to a pair of op and arguments (will be added/override the arguments provided in __call__() function).
                             For given value in a sample dict, it will look for the first match in the order dict and will apply the op on this specific key.
                             The ops specified in patterns_dict, must implement a __call__ method with an argument called key.
        """
        super().__init__()
        self._patterns_dict = Patterns(patterns_dict, (None, None))

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        See super class
        """

        for key in sample_dict.keypaths():
            op, op_kwargs_to_add = self._patterns_dict.get_value(key)
            if op is None:
                continue

            op_kwargs = copy.copy(kwargs)
            op_kwargs.update(op_kwargs_to_add)
            sample_dict = op_call(op, sample_dict, f"{op_id}_{key}", key=key, **op_kwargs)

            assert not isinstance(
                sample_dict, list
            ), f"splitting samples within {type(self).__name__} operation is not supported"

            if sample_dict is None:
                return None
            elif not isinstance(sample_dict, dict):
                raise Exception(f"unexpected sample_dict type {type(sample_dict)}")

        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        See super class
        """
        op, _ = self._patterns_dict.get_value(key_to_follow)
        if op is None:
            return

        sample_dict = op_reverse(op, sample_dict, key_to_reverse, key_to_follow, f"{op_id}_{key_to_follow}")

        return sample_dict


class OpApplyTypes(OpReversibleBase):
    """
    Select and apply an operation according value type (inferred from key name). See OpBase for more information about how it is inferred.
    Instead of specifying every relevant key, the op will be applied for every key that matched a specified pattern
    Example:
    types_dict = {  DataType.Image: (op_affine_image, dict()),
                    DataType.Seg: (op_affine_image, dict()),
                     BBox: (op_affine_bbox, dict())}

    op_apply_type = OpApplyTypes(types_dict)
    """

    def __init__(self, type_to_op_dict: Dict[Enum, Tuple[OpBase, dict]], type_detector: TypeDetectorBase):
        """
        :param type_to_op_dict: map a type (See enum DataType) to a pair of op and correspending arguments (will be added/override the arguments provided in __call__() function)
        """
        super().__init__()
        self._type_to_op_dict = type_to_op_dict
        self._type_detector = type_detector

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        See super class
        """
        all_keys = sample_dict.keypaths()
        for key in all_keys:
            key_type = self._type_detector.get_type(sample_dict, key)

            op, op_kwargs_to_add = self._type_to_op_dict.get(key_type, (None, None))
            if op is None:
                continue

            op_kwargs = copy.copy(kwargs)
            op_kwargs.update(op_kwargs_to_add)
            if "key" in op_kwargs:
                raise Exception(
                    'OpApplyTypes::"key" is already found in kwargs. Are you calling OpApplyTypes from within OpApplyTypes? it is not supported.'
                )
            sample_dict = op_call(op, sample_dict, f"{op_id}_{key}", key=key, **op_kwargs)

            assert not isinstance(
                sample_dict, list
            ), f"splitting samples within {type(self).__name__} operation is not supported"

            if sample_dict is None:
                return None
            elif not isinstance(sample_dict, dict):
                raise Exception(f"unexpected sample_dict type {type(sample_dict)}")

        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        See super class
        """
        key_type = self._type_detector.get_type(sample_dict, key_to_follow)
        op, _ = self._type_to_op_dict.get(key_type, (None, None))
        if op is None:
            return

        sample_dict = op_reverse(op, sample_dict, key_to_reverse, key_to_follow, f"{op_id}_{key_to_follow}")

        return sample_dict


class OpCollectMarker(OpReversibleBase):
    """
    Use this op within the dynamic pipeline to optimize the reading time for components such as sampler, export and stats that don't need to read the entire sample.
    OpCollectMarker will specify the last op to call to get all the required information from sample.
    In addition, to avoid from reading the entire sample including images, OpCollectMarker can also specify the list of keys required for the relevant part of the dynamic pipeline.

    Examples:
    1.
    The static pipeline generates a sample including an image ('data.image') and a label ('data.label').
    The training set sampler configured to balance a batch according to 'data.label'
    To optimize the reading time of the sampler:
    Add at the beginning of the dynamic pipeline -
    OpCollectMarker(name="sampler", static_keys_deps=["data.label"])
    2.
    The static pipeline generate an image ('data.image') and a metadata ('data.metadata').
    The dynamic pipeline includes few operations reading 'data.metadata' and that set a value used to balance the class (op_do and op_convert).
    To optimize the reading time of the sampler:
    Move op_do and op_convert to the beginning of the pipeline.
    Add just after them the following op:
    OpCollectMarker(name="sampler", static_kets_deps=["data.metadata"])

    In both cases the sampler can now read subset of the sample using: dataset.get_multi(collect_marker_name="sampler", ..)
    """

    def __init__(self, name: str, static_key_deps: Sequence[str]):
        super().__init__()
        self._name = name
        self._static_keys_deps = static_key_deps

    def get_info(self) -> dict:
        """
        Returns collect marker info including name and static_keys_deps
        """
        return {"name": self._name, "static_keys_deps": self._static_keys_deps}

    def __call__(self, sample_dict: dict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict


class OpKeepKeypaths(OpBase):
    """
    Use this op to keep only the defined keypaths in the sample
    A case where this is useful is if you want to limit the amount of data that gets transferred by multiprocessing by DataLoader workers.
    You can keep only what you want to enter the collate.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, keep_keypaths: List[str]) -> Union[None, dict, List[dict]]:
        prev_sample_dict = sample_dict
        sample_dict = NDict()
        for k in keep_keypaths:
            sample_dict[k] = prev_sample_dict[k]

        return sample_dict


class OpDeleteKeypaths(OpBase):
    """
    Use this op to remove keypaths from the sample
    A case where this is useful is if you want to limit the amount of data that gets transferred by multiprocessing by DataLoader workers.
    """

    def __call__(self, sample_dict: NDict, keypaths: List[str]) -> Union[None, dict, List[dict]]:
        for k in keypaths:
            del sample_dict[k]

        return sample_dict


class OpLookup(OpBase):
    """
    Convert a value to another value. It should be specified in a dictionary mapping old value to a new value
    Example:
    To read the gender represented by strings "male" and "female" and convert it to int do the following
    (OpLookup(map={"male": 0, "female": 1}). dict(key_in="data.input.gender", key_out="data.input.gender"))
    """

    def __init__(self, map: dict, not_exist_error: bool = True):
        """
        :param not_exist_error: false iff if the value does not exist it will keep the previous value
        """
        super().__init__()
        self._map = map
        self._not_exist_error = not_exist_error

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> Union[None, dict, List[dict]]:
        """
        :param key_in: key to a value
        :param key_out: key to store the converted value
        """
        value = sample_dict[key_in]
        if value in self._map:
            sample_dict[key_out] = self._map[value]
        elif self._not_exist_error:
            raise Exception(f"value {value} does not exist in mapping")

        return sample_dict


class OpToOneHot(OpBase):
    """
    Map category value to one hot vector
    """

    def __init__(self, num_classes: int):
        """
        :param num_classes: the size of the one hot vector
        """
        super().__init__()
        self._num_classes = num_classes

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> Union[None, dict, List[dict]]:
        """
        :param key_in: key to a class number (int)
        :param key_out: key to store the one hot vector
        """
        value = sample_dict[key_in]
        one_hot = np.zeros(self._num_classes)
        one_hot[value] = 1.0
        sample_dict[key_out] = one_hot

        return sample_dict


class OpConcat(OpBase):
    """
    Concatenate list of numpy arrays along a given axis.
    To create clinical vector that includes all the clinical information about a patient and save into "data.input.clinical" do:
    (OpConcat(), dict(keys_int=["data.input.age", "data.input.gender_one_hot", "data.input.smoking_history"], key_out="data.input_clinical", axis=0)
    """

    def __call__(
        self, sample_dict: NDict, keys_in: Sequence[str], key_out: str, axis: int = 0
    ) -> Union[None, dict, List[dict]]:
        """
        :param keys_in: sequence of keys to numpy arrays we want to concatenate
        :param key_out: the key to store the concatenated vector
        :param axis: concatenate along the specified axis
        """
        values = [np.asarray(sample_dict[key_in]) for key_in in keys_in]
        values = [v if len(v.shape) > 0 else np.expand_dims(v, axis=0) for v in values]
        sample_dict[key_out] = np.concatenate(values, axis=axis)

        return sample_dict


class OpOverrideNaN(OpBase):
    """
    Override missing values (value equals to nan)
    """

    def __call__(self, sample_dict: NDict, key: str, value_to_fill: Any) -> NDict:
        assert key in sample_dict, f"Error: missing {key}, available keys {sample_dict.keypaths()} "
        if isinstance(sample_dict[key], numbers.Number) and math.isnan(sample_dict[key]):
            sample_dict[key] = value_to_fill
        return sample_dict


class OpZScoreNorm(OpBase):
    def __call__(self, sample_dict: NDict, key: str, mean: float, std: float):
        sample_dict[key] = (sample_dict[key] - mean) / std
        return sample_dict


class OpCond(OpBase):
    """Apply given op if the condition (either directly specified or read from the sample_dict) is True"""

    def __init__(self, op: OpBase):
        """
        :param op: the op to apply
        """
        super().__init__()
        self._op = op

    def __call__(self, sample_dict: NDict, condition: Union[str, bool], **kwargs) -> Union[None, dict, List[dict]]:
        """
        :param condition:instruct if to call the inner op. Can either a boolean or a key to sample_dict used to extract the boolean
        """
        if isinstance(condition, str):
            condition = sample_dict[condition]
        if condition:
            return self._op(sample_dict, **kwargs)
        else:
            return sample_dict
