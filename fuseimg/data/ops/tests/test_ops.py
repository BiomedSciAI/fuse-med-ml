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

    # FIXME: visualizer
    # def test_basic_show(self):
    #     """
    #     Test standard backward and forward pipeline
    #     """

    #     sample = TestOps.create_sample_1(views=1)
    #     visual = Imaging2dVisualizer()
    #     VProbe = partial(VisProbe,
    #                      keys=  ["data.viewpoint1.img", "data.viewpoint1.seg" ],
    #                      type_detector=type_detector_imaging,
    #                      visualizer = visual, cache_path="~/")
    #     show_flags = VisFlag.COLLECT | VisFlag.FORWARD | VisFlag.ONLINE

    #     image_downsample_factor = 0.5
    #     pipeline = PipelineDefault('test_pipeline', [
    #         (OpRepeat(OpLoadImage(), [
    #             dict(key_in = 'data.viewpoint1.img_filename', key_out='data.viewpoint1.img'),
    #             dict(key_in = 'data.viewpoint1.seg_filename', key_out='data.viewpoint1.seg')]), {}),
    #         (op_select_slice, {"slice_idx": 50}),
    #         (op_to_int_image_space, {} ),
    #         (op_draw_grid, {"grid_size": 50}),
    #         (VProbe(flags=VisFlag.SHOW_ALL_COLLECTED | VisFlag.FORWARD|VisFlag.REVERSE|VisFlag.ONLINE), {}),
    #         (OpSample(OpAffineTransform2D(do_image_reverse=True)), {
    #             'auto_center' : True,
    #             'output_safety_size_rel': 2.0, #this is only the buffer
    #             'final_scale': image_downsample_factor,
    #             'rotate':  Uniform(-180.0,360.0),    #double range (was middle of range originaly)    #-6.0,12.0],                    #['dist@uniform',-90.0,180.0],           #uniform(-90.0, 180.0),
    #             'resampling_api': 'cv',
    #             'zoom':  Uniform(1.0,0.5),           #uniform(1.0, 0.1), 1.0,
    #             'translate_rel_pre' : 0.0, #['dist@uniform',0.0,0.05],           #uniform(0.0,0.05),
    #             #'interp' : 'linear', #1 is linear, 0 is nearest - notice - nearest may have a problem in opencv resampling_api
    #             'interp': 'linear', #Choice(['linear','nearest']),
    #             'flip_lr': RandBool(0.5)}),
    #         (OpCropNonEmptyAABB(), {}),
    #         (VProbe( flags=VisFlag.COLLECT | VisFlag.FORWARD | VisFlag.ONLINE), {}),
    #         # (OpSample(op_gamma), dict(gamma=Uniform(0.8,1.2), gain=Uniform(0.9,1.1), clip=(0,1))),
    #     ])

    #     sample = pipeline(sample)
    #     rev = pipeline.reverse(sample, key_to_follow='data.viewpoint1.img', key_to_reverse='data.viewpoint1.img')


if __name__ == "__main__":
    unittest.main()
