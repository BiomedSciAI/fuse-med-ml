from typing import List, Optional, Sequence, Union


from fuse.utils.rand.param_sampler import RandBool, draw_samples_recursively

from fuse.data.ops.op_base import OpBase, OpReversibleBase, op_call, op_reverse
from fuse.data.ops.ops_common import OpRepeat

from fuse.utils.ndict import NDict


class OpRandApply(OpReversibleBase):
    def __init__(self, op: OpBase, probability: float):
        """
        Randomly apply the op (according to the given probability)
        :param op: op
        """
        super().__init__()
        self._op = op
        self._param_sampler = RandBool(probability=probability)

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        See super class
        """
        apply = self._param_sampler.sample()
        sample_dict[op_id] = apply
        if apply:
            sample_dict = op_call(self._op, sample_dict, f"{op_id}.apply", **kwargs)
        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        See super class
        """
        apply = sample_dict[op_id]
        if apply:
            sample_dict = op_reverse(self._op, sample_dict, key_to_reverse, key_to_follow, f"{op_id}.apply")

        return sample_dict


class OpSample(OpReversibleBase):
    """
    recursively searches for ParamSamplerBase instances in kwargs, and replaces the drawn values inplace before calling to op.__call__()

    For example:
    from fuse.utils.rand.param_sampler import Uniform
    pipeline_desc = [
        #...
        OpSample(OpRotateImage()), {'rotate_angle': Uniform(0.0,360.0)}
        #...
    ]

    OpSample will draw from the Uniform distribution, and will (e.g.) pass rotate_angle=129.43 to OpRotateImage call.

    """

    def __init__(self, op: OpBase):
        """
        :param op: op
        """
        super().__init__()
        self._op = op

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        """
        See super class
        """
        sampled_kwargs = draw_samples_recursively(kwargs)
        return op_call(self._op, sample_dict, op_id, **sampled_kwargs)

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        """
        See super class
        """
        return op_reverse(self._op, sample_dict, key_to_reverse, key_to_follow, op_id)


class OpSampleAndRepeat(OpSample):
    """
    First sample kwargs and then repeat op with the exact same sampled arguments.
    This is the equivalent of using OpSample around an OpRepeat.

    Typical usage pattern:
    pipeline_desc = [
        (OpSampleAndRepeat(
            [op to run],
            [a list of dicts describing what to repeat] ),
            [a dictionary describing values that should be the same in all repeated invocations, may include sampling operations like Uniform, RandBool, etc.] ),
    ]

    Example use case:
        randomly choose a rotation angle, and then use the same randomly selected rotation angle
         for both an image and its respective ground truth segmentation map

    from fuse.utils.rand.param_sampler import Uniform
    pipeline_desc = [
        #...
        (OpSampleAndRepeat(OpRotateImage(),
            [dict(key='data.input.img'), dict(key='data.gt.seg')] ),
            dict(angle=Uniform(0.0,360.0)) #this will be drawn only once and the same value will be passed on both OpRotateImage invocation
        ),
        #...
    ]

    #note: this is a convenience op, and it is the equivalent of composing OpSample and OpRepeat yourself.
    The previous example is effectively the same as:

    pipeline_desc = [
        #...
        OpSample(OpRepeat(OpRotateImage(
            [dict(key='data.input.img'), dict(key='data.gt.seg')]),
            dict(angle=Uniform(0.0,360.0)))
        ),
        #...
    ]

    note: see OpRepeatAndSample if you are searching for the opposite flow - drawing a different value per repeat invocation
    """

    def __init__(self, op: OpBase, kwargs_per_step_to_add: Sequence[dict]):
        """
        :param op: the operation to repeat with the same sampled arguments
        :param kwargs_per_step_to_add: sequence of arguments (kwargs format) specific for a single repetition. those arguments will be added/overide the kwargs provided in __call__() function.
        """
        super().__init__(OpRepeat(op, kwargs_per_step_to_add))


class OpRepeatAndSample(OpRepeat):
    """
    Repeats an op multiple times, each time with different kwargs, and draws random values from distribution SEPARATELY per invocation.

    An example usage scenario, let's say that you train a model which is expected get as input two images:
    'data.input.adult_img' which is an image of an adult, and
    'data.input.child_img' which is an image of a child

    the model task is to predict if this child is a child of this adult (a binary classification task).

    The model is expected to work on images that are rotated to any angle, and there's no reason to suspect correlation between the rotation of the two images,
    so you would like to use rotation augmentation separately for the two images.

    In this case you could do:

    pipeline_desc = [
        #...
        (OpRepeatAndSample(OpRotateImage(),
            [dict(key='data.input.adult_img'), dict(key='data.input.child_img')]),
            dict(dict(angle=Uniform(0.0,360.0))   ### this will be drawn separately per OpRotateImage invocation
        )
        #...
    ]


    note: see also OpSampleAndRepeat if you are looking for the opposite flow, drawing the same value and using it for all repeat invocations
    """

    def __init__(self, op: OpBase, kwargs_per_step_to_add: Sequence[dict]):
        """
        :param op: the operation to repeat
        :param kwargs_per_step_to_add: sequence of arguments (kwargs format) specific for a single repetition. those arguments will be added/overide the kwargs provided in __call__() function.
        """
        super().__init__(OpSample(op), kwargs_per_step_to_add)
