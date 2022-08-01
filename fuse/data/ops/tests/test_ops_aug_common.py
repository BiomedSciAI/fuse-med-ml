import unittest

from typing import Optional, Union, List
from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase, OpReversibleBase, op_call, op_reverse
from fuse.data import create_initial_sample
from fuse.data import OpRepeat
from fuse.data.ops.ops_aug_common import OpRandApply, OpSample, OpSampleAndRepeat, OpRepeatAndSample
from fuse.utils.rand.param_sampler import Choice, RandBool, RandInt, Uniform
from fuse.utils import Seed


class OpArgsForTest(OpReversibleBase):
    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, op_id: Optional[str], **kwargs) -> Union[None, dict, List[dict]]:
        return {"op_id": op_id, "kwargs": kwargs}

    def reverse(self, sample_dict: NDict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return {"op_id": op_id}


class OpBasicSetter(OpBase):
    """
    A basic op for testing, which sets sample_dict[key] to set_key_to_val
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key, set_key_to_val, **kwargs) -> Union[None, dict, List[dict]]:
        sample_dict[key] = set_key_to_val
        return sample_dict


class TestOpsAugCommon(unittest.TestCase):
    def test_op_sample(self):
        Seed.set_seed(0)
        a = {
            "a": 5,
            "b": [3, RandInt(1, 5), 9],
            "c": {"d": 3, "f": [1, 2, RandBool(0.5), {"h": RandInt(10, 15)}]},
            "e": {"g": Choice([6, 7, 8])},
        }
        op = OpSample(OpArgsForTest())
        result = op_call(op, {}, "op_id", **a)
        b = result["kwargs"]
        call_op_id = result["op_id"]
        # make sure the same op_id passed to internal op
        self.assertEqual(call_op_id, "op_id")

        # make srgs sampled correctly
        self.assertEqual(a["a"], a["a"])
        self.assertEqual(b["b"][0], a["b"][0])
        self.assertEqual(b["b"][2], a["b"][2])
        self.assertEqual(b["c"]["d"], a["c"]["d"])
        self.assertEqual(b["c"]["f"][1], a["c"]["f"][1])
        self.assertIn(b["b"][1], [1, 2, 3, 4, 5])
        self.assertIn(b["c"]["f"][2], [True, False])
        self.assertIn(b["c"]["f"][3]["h"], [10, 11, 12, 13, 14, 15])
        self.assertIn(b["e"]["g"], [6, 7, 8])

        # make sure the same op_id passed also in reverse
        result = op_reverse(op, {}, "", "", "op_id")
        reversed_op_id = result["op_id"]
        self.assertEqual(reversed_op_id, "op_id")

    def test_op_sample_and_repeat(self):
        Seed.set_seed(1337)
        sample_1 = create_initial_sample(0)
        op = OpSampleAndRepeat(OpBasicSetter(), [dict(key="data.input.img"), dict(key="data.gt.seg")])
        sample_1 = op_call(op, sample_1, op_id="testing_sample_and_repeat", set_key_to_val=Uniform(3.0, 6.0))

        Seed.set_seed(1337)
        sample_2 = create_initial_sample(0)
        op = OpSample(OpRepeat(OpBasicSetter(), [dict(key="data.input.img"), dict(key="data.gt.seg")]))
        sample_2 = op_call(op, sample_2, op_id="testing_sample_and_repeat", set_key_to_val=Uniform(3.0, 6.0))

        self.assertEqual(sample_1["data.input.img"], sample_1["data.gt.seg"])
        self.assertEqual(sample_1["data.input.img"], sample_2["data.input.img"])

    def test_op_repeat_and_sample(self):
        Seed.set_seed(1337)
        sample_1 = create_initial_sample(0)
        op = OpRepeatAndSample(OpBasicSetter(), [dict(key="data.input.img"), dict(key="data.gt.seg")])
        sample_1 = op_call(op, sample_1, op_id="testing_sample_and_repeat", set_key_to_val=Uniform(3.0, 6.0))

        Seed.set_seed(1337)
        sample_2 = create_initial_sample(0)
        op = OpRepeat(
            OpSample(
                OpBasicSetter(),
            ),
            [dict(key="data.input.img"), dict(key="data.gt.seg")],
        )
        sample_2 = op_call(op, sample_2, op_id="testing_sample_and_repeat", set_key_to_val=Uniform(3.0, 6.0))

        self.assertEqual(sample_1["data.input.img"], sample_2["data.input.img"])
        self.assertEqual(sample_1["data.gt.seg"], sample_2["data.gt.seg"])

    def test_op_rand_apply(self):
        """
        Test OpRandApply
        """
        Seed.set_seed(0)
        op = OpRandApply(OpArgsForTest(), 0.5)

        def sample(op):
            return "kwargs" in op_call(op, {}, "op_id", a=5)

        # test range
        self.assertIn(sample(op), [True, False])

        # test generate more than a single number
        Seed.set_seed(0)
        values = [sample(op) for _ in range(4)]
        self.assertIn(True, values)
        self.assertIn(False, values)

        # test probs
        Seed.set_seed(0)
        op = OpRandApply(OpArgsForTest(), 0.99)
        count = 0
        for _ in range(1000):
            if sample(op) == True:
                count += 1
        self.assertGreaterEqual(count, 980)


if __name__ == "__main__":
    unittest.main()
