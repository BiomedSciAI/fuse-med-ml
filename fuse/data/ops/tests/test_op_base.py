import unittest

from typing import Union, List
from fuse.utils.ndict import NDict

from fuse.data.ops.op_base import OpBase, op_call
from fuse.data.key_types import DataTypeBasic
from fuse.data import create_initial_sample
from fuse.data.key_types_for_testing import DataTypeForTesting, type_detector_for_testing


class TestOpBase(unittest.TestCase):
    def test_for_type_detector(self):
        td = type_detector_for_testing
        sample = create_initial_sample("dummy")

        self.assertEqual(td.get_type(sample, "data.cc.img_for_testing"), DataTypeForTesting.IMAGE_FOR_TESTING)
        self.assertEqual(td.get_type(sample, "data.cc_img_for_testing"), DataTypeForTesting.IMAGE_FOR_TESTING)
        self.assertEqual(td.get_type(sample, "data.img_seg_for_testing"), DataTypeForTesting.SEG_FOR_TESTING)
        self.assertEqual(td.get_type(sample, "data.imgseg_for_testing"), DataTypeForTesting.SEG_FOR_TESTING)
        self.assertEqual(td.get_type(sample, "data"), DataTypeBasic.UNKNOWN)
        self.assertEqual(td.get_type(sample, "bbox_for_testing"), DataTypeForTesting.BBOX_FOR_TESTING)
        self.assertEqual(td.get_type(sample, "a.bbox_for_testing"), DataTypeForTesting.BBOX_FOR_TESTING)

    def test_op_base(self):
        class OpImp(OpBase):
            def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:
                sample_dict["data.cc.seg_for_testing"] = 5
                return sample_dict

        op = OpImp()
        sample_dict = {}
        sample_dict = op_call(op, sample_dict, "id")
        self.assertTrue("data.cc.seg_for_testing" in sample_dict)
        self.assertTrue(sample_dict["data.cc.seg_for_testing"] == 5)
        self.assertTrue(
            type_detector_for_testing.get_type(sample_dict, "data.cc.seg_for_testing")
            == DataTypeForTesting.SEG_FOR_TESTING
        )
        type_detector_for_testing.verify_type(
            sample_dict, "data.cc.seg_for_testing", [DataTypeForTesting.SEG_FOR_TESTING]
        )
        self.assertRaises(
            ValueError,
            type_detector_for_testing.verify_type,
            sample_dict,
            "data.cc.seg_for_testing",
            [DataTypeForTesting.IMAGE_FOR_TESTING],
        )


if __name__ == "__main__":
    unittest.main()
