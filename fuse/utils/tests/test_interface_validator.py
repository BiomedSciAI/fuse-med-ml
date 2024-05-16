import unittest
from fuse.utils.interface_validator import InterfaceValidator, validate_signature
from typing import Dict


class TestInterfaceValidator(unittest.TestCase):
    def test_1(self) -> None:
        class FruitBase(InterfaceValidator):
            def __init__(self) -> None:
                super().__init__()

            @validate_signature(
                args=["color"]
            )  # only argument `color` will be validated to both exist and have the same type hint in the inheriting class
            def draw(
                self, *, color: str = "red", style: int = 12, **kwargs: Dict
            ) -> None:
                print(f"FruitBase.draw: color={color} style={style}")

            @validate_signature(
                args=["size", "material"]
            )  # both args `size` and `material` will be validated
            def print_3d(self, *, size: float = 2.7, material: str = "iron") -> None:
                print(f"FruitBase.print_3d size={size} material={material}")

        class TestClass1_missing_function(FruitBase):
            def __init__(self, *, arg: str) -> None:
                super().__init__()
                self.arg = arg

            def draw(
                self, *, blah: str, color: str = "red", hoho: str = "hoho"
            ) -> None:
                pass

            # we don't have print_3d method

        class TestClass1_no_super_init(FruitBase):
            def __init__(self, *, arg: str) -> None:
                # we forgot to call super().__init__() here
                self.arg = arg

            def draw(
                self, *, color: str = "red", style: int = 12, **kwargs: Dict
            ) -> None:
                print(f"FruitBase.draw: color={color} style={style}")

            def print_3d(self, *, size: float = 2.7, material: str = "iron") -> None:
                print(f"FruitBase.print_3d size={size} material={material}")

        class TestClass1_with_super_init_but_incorrect_method_args(FruitBase):
            def __init__(self, *, arg: str) -> None:
                super().__init__()
                self.arg = arg

            def draw(
                self, *, coco_color: str = "red", style: int = 12, **kwargs: Dict
            ) -> None:
                print(f"FruitBase.draw: coco_color={coco_color} style={style}")

            def print_3d(
                self, *, size: float = 2.7, momo_material: str = "iron"
            ) -> None:
                print(f"FruitBase.print_3d size={size} momo_material={momo_material}")

        class TestClass1_with_incorrect_type_hints(FruitBase):
            def __init__(self, *, arg: str) -> None:
                super().__init__()
                self.arg = arg

            # color should be type hint str, but we change it to int to see that the interface validator catches it
            def draw(self, *, color: int = 2, style: int = 12, **kwargs: Dict) -> None:
                print(f"FruitBase.draw: color={color} style={style}")

            def print_3d(self, *, size: float = 2.7, material: str = "iron") -> None:
                print(f"FruitBase.print_3d size={size} material={material}")

        class TestClass1_with_pos_args(FruitBase):
            def __init__(self, *, arg: str) -> None:
                super().__init__()
                self.arg = arg

            def draw(
                self, boo: str, *, color: str = "red", style: int = 12, **kwargs: Dict
            ) -> None:
                print(f"FruitBase.draw: color={color} style={style}")

            def print_3d(self, *, size: float = 2.7, material: str = "iron") -> None:
                print(f"FruitBase.print_3d size={size} material={material}")

        class TestClass1_with_correct_args(FruitBase):
            def __init__(self, *, arg: str) -> None:
                super().__init__()
                self.arg = arg

            def draw(
                self, *, color: str = "red", style: int = 12, **kwargs: Dict
            ) -> None:
                print(f"FruitBase.draw: color={color} style={style}")

            def print_3d(self, *, size: float = 2.7, material: str = "iron") -> None:
                print(f"FruitBase.print_3d size={size} material={material}")

        self.assertRaises(Exception, lambda: TestClass1_missing_function(arg="banana"))
        self.assertRaises(Exception, lambda: TestClass1_no_super_init(arg="banana"))
        self.assertRaises(
            Exception,
            lambda: TestClass1_with_super_init_but_incorrect_method_args(arg="banana"),
        )
        self.assertRaises(
            Exception, lambda: TestClass1_with_incorrect_type_hints(arg="banana")
        )
        self.assertRaises(Exception, lambda: TestClass1_with_pos_args(arg="banana"))

        inst = TestClass1_with_correct_args(arg="banana")


if __name__ == "__main__":
    unittest.main()
