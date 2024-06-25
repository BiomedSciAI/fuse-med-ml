from typing import List, Dict, Callable, Any  # , _UnionGenericAlias


import inspect

# https://stackoverflow.com/questions/147816/preserving-signatures-of-decorated-functions
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
# https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class


"""
This is a very handy class that allows enforcing an API
Unlike programming languages like C++, python does not have a built in way to enforce adherance to API when inheriting from a base class
By using the code found in this script you can require the inheriting class to implement certain method and also control which arguments of the method signature is required.
an actively used example can be seen in InsilicoLabModelBase() in insilico-lab-validation repo.
For illustration, you can create a base class FruitBase (you can see related unit test in `fuse-med-ml/fuse/utils/tests/test_interface_validator.py`)


```python
from fuse.utils.interface_validator import InterfaceValidator, validate_signature

class FruitBase(InterfaceValidator):
    def __init__(self, *, wow='wow'):
        super().__init__()
        self.wow = wow

    @validate_signature(args=['color'])  #only argument `color` will be validated to both exist and have the same type hint in the inheriting class
    def draw(self, *, color:str = 'red', style:int=12, **kwargs):
        print(f'FruitBase.draw: {color=} {style=}')

    @validate_signature(args=['size', 'material']) #both args `size` and `material` will be validated
    def print_3d(self, *, size:float = 2.7, material:str = 'iron'):
        print(f'FruitBase.print_3d {size=} {material=}')
```

now, when you use a class that inherits from it, like:

```python
class Banana(FruitBase):
    #...
```

the API will be enforced and exceptions will be raised if requirements are not met.


#TODO:
    1. allow validating return type hint
    2. allow using @validate_signature without using any args, and that would mean that the exact same full signature is required
    3. allow control over whether type hint is validated or not (is this needed?)
"""


def validate_signature(*, args: List[str] = None) -> Callable:
    def wrapper(function: Callable) -> Any:
        return validate_signature_impl(function, args=args)

    return wrapper


class validate_signature_impl(object):
    def __init__(self, fn: Callable, args: List[str] = None) -> None:
        self.fn = fn
        self._args_to_validate = args

    def __set_name__(self, owner: Any, name: str) -> None:
        # do something with owner, i.e.
        curr_func = self.fn
        print(f"decorating {curr_func} and using {owner}")
        if not hasattr(owner, "_validate_methods"):
            owner._validate_methods = {}
        owner._validate_methods[name] = {}

        signature_info = inspect.getfullargspec(self.fn)

        for curr_arg in self._args_to_validate:
            owner._validate_methods[name][curr_arg] = signature_info.annotations[
                curr_arg
            ]

        # then replace ourself with the original method
        setattr(owner, name, curr_func)


class InterfaceValidator:
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.__dict__["_properly_initialized_InterfaceValidator"] = True
        self._verbose_signature_validation = verbose
        if not hasattr(self, "_validate_methods"):
            raise Exception(
                "InterfaceValidator: error - seems like you did not requested signature validation for any method. To use that you need to use @interface_validator.validate_signature on at least one method"
            )

        for method_name, curr_validate_args in self._validate_methods.items():
            curr_bases = self.__class__.__bases__
            if len(curr_bases) > 1:
                raise Exception("currently not supporting more than one base")
            if len(curr_bases) == 0:
                raise Exception("expected at least one base")

            if getattr(type(self), method_name) == getattr(curr_bases[0], method_name):
                raise Exception(
                    f'a class that inherited from InterfaceValidator did not override interface method "{method_name}" !'
                )

            curr_method = getattr(self, method_name)
            if self._verbose_signature_validation:
                print(
                    f'found method "{curr_method}" with requested validation of signature for args {curr_validate_args}'
                )
            self._validate_signature(method_name, curr_validate_args)

    def _validate_signature(
        self, method_name: str, validate_args: Dict[str, Any]
    ) -> None:
        """
        raises an exception if a signature mismatch is found
        """
        curr_method = getattr(self, method_name)
        actual_method = curr_method

        func_details = inspect.getfullargspec(actual_method)

        pos_args = [x for x in func_details.args if x != "self"]
        if len(pos_args) > 0:
            raise Exception(
                f'found positional args in the function of {actual_method} - this is not supported. Positional args found = {pos_args}. Tip: you can put * after the "self" argument to prevent positional args. For example: def foo(self, *, arg1:int, arg2:float=2.3) '
            )

        for arg_name, arg_type_hint in validate_args.items():
            if arg_name not in func_details.kwonlyargs:
                raise Exception(
                    f'error in class {self.__class__.__name__} method {method_name} - could not find expected kwarg "{arg_name}"'
                )

            if arg_name not in func_details.annotations:
                raise Exception(
                    f"type annotation is required in class {self.__class__.__name__} in method {method_name} - arg {arg_name} "
                )

            actual_type_hint = func_details.annotations[arg_name]
            if actual_type_hint != arg_type_hint:
                raise Exception(
                    f"error in class {self.__class__.__name__} in method {method_name} - arg {arg_name} is expected to have type hint = {arg_type_hint} but got {actual_type_hint} instead"
                )
            if self._verbose_signature_validation:
                print(
                    f"in class {self.__class__.__name__} method {method_name} arg {arg_name} validated ok."
                )

    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self, "_properly_initialized_InterfaceValidator"):
            raise Exception(
                "When you inherit from InterfaceValidator you must call super().__init__() in your constructor as the first thing you do."
            )
        super().__setattr__(name, value)
