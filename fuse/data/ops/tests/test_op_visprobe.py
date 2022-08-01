import unittest

from typing import Any, Union, List
import copy
from functools import partial

from fuse.utils.ndict import NDict

from fuse.data.ops.ops_visprobe import VisFlag, VisProbe
from fuse.data.visualizer.visualizer_base import VisualizerBase
from fuse.data.ops.op_base import OpBase, OpReversibleBase
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.key_types_for_testing import type_detector_for_testing


class OpSetForTest(OpReversibleBase):
    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, op_id: str, key: str, val: Any) -> Union[None, dict, List[dict]]:
        # store information for reverse operation
        sample_dict[f"{op_id}.key"] = key
        if key in sample_dict:
            prev_val = sample_dict[key]
            sample_dict[f"{op_id}.prev_val"] = prev_val

        # set
        sample_dict[key] = val
        return sample_dict

    def reverse(self, sample_dict: NDict, op_id: str, key_to_reverse: str, key_to_follow: str) -> dict:
        key = sample_dict[f"{op_id}.key"]
        if key == key_to_follow:
            if f"{op_id}.prev_val" in sample_dict:
                prev_val = sample_dict[f"{op_id}.prev_val"]
                sample_dict[key_to_reverse] = prev_val
            else:
                if key_to_reverse in sample_dict:
                    sample_dict.pop(key_to_reverse)
        return sample_dict


class DebugVisualizer(VisualizerBase):
    acc = []

    def __init__(self) -> None:
        super().__init__()

    def _show(self, vis_data):
        if issubclass(type(vis_data), dict):
            DebugVisualizer.acc.append([vis_data])
        else:
            DebugVisualizer.acc.append(vis_data)


testing_img_key = "img_for_testing"
testing_seg_key = "seg_for_testing"
g1_testing_image_key = "data.test_pipeline." + testing_img_key
g1_testing_seg_key = "data.test_pipeline." + testing_seg_key
g2_testing_image_key = "data.test_pipeline2." + testing_img_key
g2_testing_seg_key = "data.test_pipeline2." + testing_seg_key

VProbe = partial(
    VisProbe,
    keys=[g1_testing_image_key],
    type_detector=type_detector_for_testing,
    visualizer=DebugVisualizer(),
    cache_path="~/",
)


class TestVisProbe(unittest.TestCase):
    def test_basic_show(self):
        """
        Test standard backward and forward pipeline
        """
        global g1_testing_image_key
        show_flags = VisFlag.SHOW_CURRENT | VisFlag.FORWARD | VisFlag.ONLINE
        pipeline_seq = [
            (OpSetForTest(), dict(key=g1_testing_image_key, val=5)),
            (VProbe(flags=show_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=6)),
            (VProbe(flags=show_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict({"sample_id": "a", "data": {"test_pipeline": {testing_img_key: 6}}})

        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict[g1_testing_image_key], 7)
        self.assertEqual(len(DebugVisualizer.acc), 2)
        self.assertEqual(len(DebugVisualizer.acc[0]), 1)
        self.assertEqual(len(DebugVisualizer.acc[1]), 1)
        g1_testing_key = g1_testing_image_key.replace(".", "_")
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{g1_testing_key}.value"], 5)
        self.assertEqual(DebugVisualizer.acc[1][0][f"groups.group0.{g1_testing_key}.value"], 6)
        DebugVisualizer.acc.clear()

    def test_multi_label(self):
        """
        Test standard backward and forward pipeline
        """

        VMProbe = partial(
            VisProbe,
            keys=[g1_testing_image_key, g1_testing_seg_key],
            type_detector=type_detector_for_testing,
            visualizer=DebugVisualizer(),
            cache_path="~/",
        )

        show_flags = VisFlag.SHOW_CURRENT | VisFlag.FORWARD | VisFlag.ONLINE
        pipeline_seq = [
            (OpSetForTest(), dict(key=g1_testing_image_key, val=5)),
            (VMProbe(flags=show_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=6)),
            (VMProbe(flags=show_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict(
            {
                "sample_id": "a",
                "data": {
                    "test_pipeline": {testing_img_key: 4, testing_seg_key: 4},
                    "test_pipeline2": {testing_img_key: 4, testing_seg_key: 4},
                },
            }
        )

        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict[g1_testing_image_key], 7)
        self.assertEqual(len(DebugVisualizer.acc), 2)
        self.assertEqual(len(DebugVisualizer.acc[0]), 1)
        self.assertEqual(len(DebugVisualizer.acc[1]), 1)
        test_image_key = g1_testing_image_key.replace(".", "_")
        test_seg_key = g1_testing_seg_key.replace(".", "_")
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{test_image_key}.value"], 5)
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{test_seg_key}.value"], 4)
        self.assertFalse("group1" in DebugVisualizer.acc[0][0]["groups"])
        self.assertEqual(DebugVisualizer.acc[1][0][f"groups.group0.{test_image_key}.value"], 6)
        self.assertEqual(DebugVisualizer.acc[1][0][f"groups.group0.{test_seg_key}.value"], 4)
        self.assertFalse("group1" in DebugVisualizer.acc[1][0])
        DebugVisualizer.acc.clear()

    def test_multi_groups(self):
        """
        Test standard backward and forward pipeline
        """

        VMProbe = partial(
            VisProbe,
            keys=[g1_testing_image_key, g2_testing_image_key],
            type_detector=type_detector_for_testing,
            visualizer=DebugVisualizer(),
            cache_path="~/",
        )

        show_flags = VisFlag.SHOW_CURRENT | VisFlag.FORWARD | VisFlag.ONLINE
        pipeline_seq = [
            (OpSetForTest(), dict(key=g1_testing_image_key, val=5)),
            (VMProbe(flags=show_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=6)),
            (VMProbe(flags=show_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict(
            {
                "sample_id": "a",
                "data": {
                    "test_pipeline": {testing_img_key: 4, testing_seg_key: 4},
                    "test_pipeline2": {testing_img_key: 4, testing_seg_key: 4},
                },
            }
        )

        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict[g1_testing_image_key], 7)
        self.assertEqual(len(DebugVisualizer.acc), 2)
        self.assertEqual(len(DebugVisualizer.acc[0]), 1)
        self.assertEqual(len(DebugVisualizer.acc[1]), 1)
        test_image_key_g1 = g1_testing_image_key.replace(".", "_")
        test_image_key_g2 = g2_testing_image_key.replace(".", "_")
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{test_image_key_g1}.value"], 5)
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group1.{test_image_key_g2}.value"], 4)
        self.assertEqual(DebugVisualizer.acc[1][0][f"groups.group0.{test_image_key_g1}.value"], 6)
        self.assertEqual(DebugVisualizer.acc[1][0][f"groups.group1.{test_image_key_g2}.value"], 4)
        DebugVisualizer.acc.clear()

    def test_collected_show(self):
        """
        Test basic collected compare
        """
        forward_flags = VisFlag.FORWARD | VisFlag.ONLINE

        pipeline_seq = [
            (OpSetForTest(), dict(key=g1_testing_image_key, val=5)),
            (VProbe(flags=forward_flags | VisFlag.COLLECT), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=6)),
            (VProbe(flags=forward_flags | VisFlag.SHOW_COLLECTED), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict({"sample_id": "a", "data": {"test_pipeline": {testing_img_key: 6}}})

        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict[g1_testing_image_key], 7)
        self.assertEqual(len(DebugVisualizer.acc), 1)
        self.assertEqual(len(DebugVisualizer.acc[0]), 2)
        test_image_key = g1_testing_image_key.replace(".", "_")
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{test_image_key}.value"], 5)
        self.assertEqual(DebugVisualizer.acc[0][1][f"groups.group0.{test_image_key}.value"], 6)
        DebugVisualizer.acc.clear()

    def test_reverse_compare(self):
        """
        Test compare of collected forward with reverse of same op
        """
        revfor_flags = VisFlag.FORWARD | VisFlag.ONLINE | VisFlag.REVERSE | VisFlag.SHOW_COLLECTED

        pipeline_seq = [
            (OpSetForTest(), dict(key=g1_testing_image_key, val=5)),
            (VProbe(flags=revfor_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=6)),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict({"sample_id": "a", "data": {"test_pipeline": {testing_img_key: 4}}})

        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict[g1_testing_image_key], 7)
        sample_dict = pipe.reverse(sample_dict, g1_testing_image_key, g1_testing_image_key)
        self.assertEqual(len(DebugVisualizer.acc), 1)
        self.assertEqual(len(DebugVisualizer.acc[0]), 2)
        test_image_key = g1_testing_image_key.replace(".", "_")
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{test_image_key}.value"], 5)
        self.assertEqual(DebugVisualizer.acc[0][1][f"groups.group0.{test_image_key}.value"], 5)
        self.assertEqual(sample_dict[g1_testing_image_key], 4)

        DebugVisualizer.acc.clear()

    def test_multiple_reverse(self):

        """
        Test compare of multiple collected forward with reverse of same op
        """
        revfor_flags = VisFlag.FORWARD | VisFlag.ONLINE | VisFlag.REVERSE | VisFlag.SHOW_COLLECTED

        pipeline_seq = [
            (OpSetForTest(), dict(key=g1_testing_image_key, val=5)),
            (VProbe(flags=revfor_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=6)),
            (VProbe(flags=revfor_flags), {}),
            (OpSetForTest(), dict(key=g1_testing_image_key, val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict({"sample_id": "a", "data": {"test_pipeline": {testing_img_key: 4}}})

        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict[g1_testing_image_key], 7)
        sample_dict = pipe.reverse(sample_dict, g1_testing_image_key, g1_testing_image_key)
        self.assertEqual(len(DebugVisualizer.acc), 2)
        self.assertEqual(len(DebugVisualizer.acc[0]), 2)
        test_image_key = g1_testing_image_key.replace(".", "_")
        self.assertEqual(DebugVisualizer.acc[0][0][f"groups.group0.{test_image_key}.value"], 6)
        self.assertEqual(DebugVisualizer.acc[0][1][f"groups.group0.{test_image_key}.value"], 6)
        self.assertEqual(DebugVisualizer.acc[1][0][f"groups.group0.{test_image_key}.value"], 5)
        self.assertEqual(DebugVisualizer.acc[1][1][f"groups.group0.{test_image_key}.value"], 5)
        self.assertEqual(sample_dict[g1_testing_image_key], 4)

        DebugVisualizer.acc.clear()

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
