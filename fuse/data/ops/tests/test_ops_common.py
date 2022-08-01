import unittest

from typing import Optional, OrderedDict, Union, List

from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase, OpReversibleBase, op_call
from fuse.data.key_types_for_testing import DataTypeForTesting

from fuse.data.ops.ops_common import OpApplyPatterns, OpFunc, OpLambda, OpRepeat
from fuse.data.ops.ops_common_for_testing import OpApplyTypesImaging


class OpIncrForTest(OpReversibleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self, sample_dict: NDict, op_id: Optional[str], incr_value: int, key_in: str, key_out: str
    ) -> Union[None, dict, List[dict]]:
        # save for reverse
        sample_dict[op_id] = {"key_out": key_out, "incr_value": incr_value}
        # apply
        value = sample_dict[key_in]
        sample_dict[key_out] = value + incr_value

        return sample_dict

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        # not really reverse, but help the test
        orig_args = sample_dict[op_id]

        if orig_args["key_out"] != key_to_follow:
            return sample_dict

        value = sample_dict[key_to_reverse]
        sample_dict[key_to_reverse] = value - orig_args["incr_value"]

        return sample_dict


class TestOpsCommon(unittest.TestCase):
    def test_op_repeat(self):
        """
        Test OpRepeat __call__() and reverse()
        """
        op_base = OpIncrForTest()
        kwargs_per_step_to_add = [
            dict(key_in="data.val.a", key_out="data.val.b"),
            dict(key_in="data.val.b", key_out="data.val.c"),
            dict(key_in="data.val.b", key_out="data.val.d"),
            dict(key_in="data.val.d", key_out="data.val.d"),
        ]
        op_repeat = OpRepeat(op_base, kwargs_per_step_to_add)
        sample_dict = NDict({})
        sample_dict["data.val.a"] = 5
        sample_dict = op_repeat(sample_dict, "_.test_repeat", incr_value=3)
        self.assertEqual(sample_dict["data.val.a"], 5)
        self.assertEqual(sample_dict["data.val.b"], 8)
        self.assertEqual(sample_dict["data.val.c"], 11)
        self.assertEqual(sample_dict["data.val.d"], 14)

        op_repeat.reverse(sample_dict, key_to_follow="data.val.d", key_to_reverse="data.val.d", op_id="_.test_repeat")
        self.assertEqual(sample_dict["data.val.a"], 5)
        self.assertEqual(sample_dict["data.val.b"], 8)
        self.assertEqual(sample_dict["data.val.c"], 11)
        self.assertEqual(sample_dict["data.val.d"], 8)

        sample_dict["data.val.e"] = 48
        op_repeat.reverse(sample_dict, key_to_follow="data.val.d", key_to_reverse="data.val.e", op_id="_.test_repeat")
        self.assertEqual(sample_dict["data.val.a"], 5)
        self.assertEqual(sample_dict["data.val.b"], 8)
        self.assertEqual(sample_dict["data.val.c"], 11)
        self.assertEqual(sample_dict["data.val.d"], 8)
        self.assertEqual(sample_dict["data.val.e"], 42)

    def test_op_lambda(self):
        """
        Test OpLambda __call__() and reverse()
        """
        op_base = OpLambda(func=lambda x: x + 3)
        kwargs_per_step_to_add = [dict(), dict(), dict()]
        op_repeat = OpRepeat(op_base, kwargs_per_step_to_add)
        sample_dict = NDict({})
        sample_dict["data.val.a"] = 5
        sample_dict = op_repeat(sample_dict, "_.test_repeat", key="data.val.a")
        self.assertEqual(sample_dict["data.val.a"], 14)

        op_base = OpLambda(func=lambda x: x + 3, func_reverse=lambda x: x - 3)
        op_repeat = OpRepeat(op_base, kwargs_per_step_to_add)
        sample_dict = NDict({})
        sample_dict["data.val.a"] = 5
        sample_dict = op_repeat(sample_dict, "_.test_repeat", key="data.val.a")
        self.assertEqual(sample_dict["data.val.a"], 14)

        op_repeat.reverse(sample_dict, key_to_follow="data.val.a", key_to_reverse="data.val.a", op_id="_.test_repeat")
        self.assertEqual(sample_dict["data.val.a"], 5)

        sample_dict["data.val.b"] = 51
        op_repeat.reverse(sample_dict, key_to_follow="data.val.a", key_to_reverse="data.val.b", op_id="_.test_repeat")
        self.assertEqual(sample_dict["data.val.a"], 5)
        self.assertEqual(sample_dict["data.val.b"], 42)

    def test_op_lambda_with_kwargs(self):
        """
        Test OpLambda __call__() with kwargs
        """
        op_base = OpLambda(func=lambda x, y: x + y)
        kwargs_per_step_to_add = [dict(), dict(), dict()]
        op_repeat = OpRepeat(op_base, kwargs_per_step_to_add)
        sample_dict = NDict()
        sample_dict["data.val.a"] = 5
        sample_dict = op_repeat(sample_dict, "_.test_repeat", key="data.val.a", y=5)
        self.assertEqual(sample_dict["data.val.a"], 20)

    def test_op_func(self):
        """
        Test OpFunc __call__()
        """

        def func_single_output(a, b, c):
            return a + b + c

        def func_multi_output(a, b, c):
            return a + b, a + c

        single_output_op = OpFunc(func=func_single_output)
        sample_dict = NDict({})
        sample_dict["data.first"] = 5
        sample_dict["data.second"] = 9
        sample_dict = op_call(
            single_output_op,
            sample_dict,
            "_.test_func",
            c=2,
            inputs={"data.first": "a", "data.second": "b"},
            outputs="data.out",
        )
        self.assertEqual(sample_dict["data.out"], 16)

        multi_output_op = OpFunc(func=func_multi_output)
        sample_dict = NDict({})
        sample_dict["data.first"] = 5
        sample_dict["data.second"] = 9
        sample_dict = op_call(
            multi_output_op,
            sample_dict,
            "_.test_func",
            c=2,
            inputs={"data.first": "a", "data.second": "b"},
            outputs=["data.out", "data.more"],
        )
        self.assertEqual(sample_dict["data.out"], 14)
        self.assertEqual(sample_dict["data.more"], 7)

    def test_op_apply_patterns(self):
        """
        Test OpRApplyPatterns __call__() and reverse()
        """

        op_add_1 = OpLambda(func=lambda x: x + 1, func_reverse=lambda x: x - 1)
        op_mul_2 = OpLambda(func=lambda x: x * 2, func_reverse=lambda x: x // 2)
        op_mul_4 = OpLambda(func=lambda x: x * 4, func_reverse=lambda x: x // 4)

        sample_dict = NDict({})
        sample_dict["data.val.img_for_testing"] = 3
        sample_dict["data.test.img_for_testing"] = 3
        sample_dict["data.test.seg_for_testing"] = 3
        sample_dict["data.test.bbox_for_testing"] = 3
        sample_dict["data.test.meta"] = 3

        patterns_dict = OrderedDict(
            [
                (r"^data.val.img_for_testing$", (op_add_1, dict())),
                (r"^.*img_for_testing$|^.*seg_for_testing$", (op_mul_2, dict())),
                (r"^data.[^.]*.bbox_for_testing", (op_mul_4, dict())),
            ]
        )
        op_apply_pat = OpApplyPatterns(patterns_dict)

        sample_dict = op_apply_pat(sample_dict, "_.test_apply_pat")
        self.assertEqual(sample_dict["data.val.img_for_testing"], 4)
        self.assertEqual(sample_dict["data.test.img_for_testing"], 6)
        self.assertEqual(sample_dict["data.test.seg_for_testing"], 6)
        self.assertEqual(sample_dict["data.test.bbox_for_testing"], 12)
        self.assertEqual(sample_dict["data.test.meta"], 3)

        sample_dict["model.seg_for_testing"] = 3
        op_apply_pat.reverse(
            sample_dict,
            key_to_follow="data.val.img_for_testing",
            key_to_reverse="model.seg_for_testing",
            op_id="_.test_apply_pat",
        )
        self.assertEqual(sample_dict["data.val.img_for_testing"], 4)
        self.assertEqual(sample_dict["model.seg_for_testing"], 2)

    def test_op_apply_types(self):
        """
        Test OpApplyTypes __call__() and reverse()
        """

        op_add_1 = OpLambda(func=lambda x: x + 1, func_reverse=lambda x: x - 1)
        op_mul_2 = OpLambda(func=lambda x: x * 2, func_reverse=lambda x: x // 2)
        op_mul_4 = OpLambda(func=lambda x: x * 4, func_reverse=lambda x: x // 4)

        sample_dict = NDict({})
        sample_dict["data.val.img_for_testing"] = 3
        sample_dict["data.test.img_for_testing"] = 3
        sample_dict["data.test.seg_for_testing"] = 3
        sample_dict["data.test.bbox_for_testing"] = 3
        sample_dict["data.test.meta"] = 3

        types_dict = {
            DataTypeForTesting.IMAGE_FOR_TESTING: (op_add_1, dict()),
            DataTypeForTesting.SEG_FOR_TESTING: (op_mul_2, dict()),
            DataTypeForTesting.BBOX_FOR_TESTING: (op_mul_4, dict()),
        }

        op_apply_type = OpApplyTypesImaging(types_dict)

        sample_dict = op_apply_type(sample_dict, "_.test_apply_type")
        self.assertEqual(sample_dict["data.val.img_for_testing"], 4)
        self.assertEqual(sample_dict["data.test.img_for_testing"], 4)
        self.assertEqual(sample_dict["data.test.seg_for_testing"], 6)
        self.assertEqual(sample_dict["data.test.bbox_for_testing"], 12)
        self.assertEqual(sample_dict["data.test.meta"], 3)

        sample_dict["model.a_seg_for_testing"] = 3
        op_apply_type.reverse(
            sample_dict,
            key_to_follow="data.val.img_for_testing",
            key_to_reverse="model.a_seg_for_testing",
            op_id="_.test_apply_type",
        )
        self.assertEqual(sample_dict["data.val.img_for_testing"], 4)
        self.assertEqual(sample_dict["model.a_seg_for_testing"], 2)


if __name__ == "__main__":
    unittest.main()
