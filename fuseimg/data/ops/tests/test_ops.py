import unittest

from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuseimg.data.ops.color import OpClip, OpToRange
from fuseimg.data.ops.shape_ops import OpPad

from fuse.utils.ndict import NDict

import numpy as np
import torch


class TestOps(unittest.TestCase):
    def test_basic_1(self):
        """
        Test basic imaging ops
        """

        sample = NDict()
        sample["data.input.img"] = np.array([5, 0.5, -5, 3])

        pipeline = PipelineDefault(
            "test_pipeline",
            [
                # (op_normalize_against_self, {} ),
                (OpClip(), dict(key="data.input.img", clip=(-0.5, 3.0))),
                (OpToRange(), dict(key="data.input.img", from_range=(-0.5, 3.0), to_range=(-3.5, 3.5))),
            ],
        )

        sample = pipeline(sample)

        self.assertLessEqual(sample["data.input.img"].max(), 3.5)
        self.assertGreaterEqual(sample["data.input.img"].min(), -3.5)
        self.assertEqual(sample["data.input.img"][-1], 3.5)

    def test_op_pad(self):
        """
        Test OpPad
        """
        sample = NDict()
        sample["data.input.tensor_img_1"] = torch.Tensor([[1]])
        sample["data.input.numpy_img_1"] = np.array([[1]])
        sample["data.input.tensor_img_2"] = torch.Tensor([[42]])
        sample["data.input.numpy_img_2"] = np.array([[42]])

        pipeline = PipelineDefault(
            "test_pipeline",
            [
                (OpPad(), dict(key="data.input.tensor_img_1", padding=1, fill=0, mode="constant")),
                (OpPad(), dict(key="data.input.numpy_img_1", padding=1, fill=0, mode="constant")),
                (OpPad(), dict(key="data.input.tensor_img_2", padding=1, fill=42, mode="constant")),
                (OpPad(), dict(key="data.input.numpy_img_2", padding=1, fill=42, mode="constant")),
            ],
        )

        pipeline(sample)

        res_1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

        res_2 = [[42, 42, 42], [42, 42, 42], [42, 42, 42]]

        self.assertTrue(np.array_equal(sample["data.input.tensor_img_1"], res_1))
        self.assertTrue(np.array_equal(sample["data.input.numpy_img_1"], res_1))
        self.assertTrue(np.array_equal(sample["data.input.tensor_img_2"], res_2))
        self.assertTrue(np.array_equal(sample["data.input.numpy_img_2"], res_2))

    def test_op_resize_to(self):
        pass


if __name__ == "__main__":
    unittest.main()
