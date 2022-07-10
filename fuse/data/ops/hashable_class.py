from typing import Callable
from fuse.data.ops import get_function_call_str
from fuse.data.ops.caching_tools import get_callers_string_description, value_to_string


class HashableClass:
    _MISSING_SUPER_INIT_ERR_MSG = "Did you forget to call super().__init__() You must call it in your __init__? Also, make sure you call it BEFORE setting any attribute."

    # controls how values are converted to string, override if you want custom behavior
    # expected signature: foo(val:Any) -> str
    VALUE_TO_STRING_FUNC: Callable = value_to_string

    def __init__(self):
        """
        when init is called, a string representation of the caller(s) init args are recorded.
        This is used in get_hashable_string_representation which is used later for hashing in caching related tools (for example, SamplesCacher)
        """
        # the following is used to extract callers args, for __init__ calls up the stack of classes inheirting from OpBase
        # this way it can happen in the base class and then anyone creating new Ops will typically only need to add
        # super().__init__ in their __init__ implementation
        self._stored_init_str_representation = get_callers_string_description(
            max_look_up=4,
            expected_class=HashableClass,
            expected_function_name="__init__",
            value_to_string_func=HashableClass.VALUE_TO_STRING_FUNC,
            ignore_first_frames=3,
        )

    def __setattr__(self, name, value):
        """
        Verifies that super().__init__() is called before setting any attribute
        """
        storage_name = "_stored_init_str_representation"
        if name != storage_name and not hasattr(self, storage_name):
            raise Exception(HashableClass._MISSING_SUPER_INIT_ERR_MSG)
        super().__setattr__(name, value)

    def get_hashable_string_representation(self) -> str:
        """
        A string representation of this operation, which will be used for hashing.
        It includes recorded (string) data describing the args that were used in __init__()
        you can override/extend it in the rare cases that it's needed.

        Note - not using __str__ or __repr__ by design, to avoid cases that developers override their Ops without being aware of the effect on caching

        example:

        class OpSomethingNew(OpBase):
            def __init__(self):
                super().__init__()
            def __str__(self):
                ans = super().__str__(self)
                ans += 'whatever you want to add"

        """

        if not hasattr(self, "_stored_init_str_representation"):
            raise Exception(HashableClass._MISSING_SUPER_INIT_ERR_MSG)
        call_repr = get_function_call_str(
            self.__call__,
        )

        return f"init_{self._stored_init_str_representation}@call_{call_repr}"
