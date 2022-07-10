import unittest

from fuse.utils.ndict import NDict
from typing import Any, Union, List
import copy
from unittest.case import expectedFailure

from fuse.data.ops.op_base import OpBase, OpReversibleBase
from fuse.data.pipelines.pipeline_default import PipelineDefault


class OpSetForTest(OpReversibleBase):
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


class OpNoneForTest(OpBase):
    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:
        return None


class OpSplitForTest(OpBase):
    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:
        sample_id = sample_dict["data.sample_id"]
        samples = []
        split_num = 10
        for index in range(split_num):
            sample = copy.deepcopy(sample_dict)
            sample["data.sample_id"] = (sample_id, index)
            samples.append(sample)

        return samples


class TestPipelineDefault(unittest.TestCase):
    def test_pipeline(self):
        """
        Test standard backward and forward pipeline
        """
        pipeline_seq = [
            (OpSetForTest(), dict(key="data.test_pipeline", val=5)),
            (OpSetForTest(), dict(key="data.test_pipeline", val=6)),
            (OpSetForTest(), dict(key="data.test_pipeline_2", val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)

        sample_dict = NDict({})
        sample_dict = pipe(sample_dict)
        self.assertEqual(sample_dict["data.test_pipeline"], 6)
        self.assertEqual(sample_dict["data.test_pipeline_2"], 7)

        sample_dict = pipe.reverse(sample_dict, "data.test_pipeline", "data.test_pipeline")
        self.assertEqual("data.test_pipeline" in sample_dict, False)
        self.assertEqual(sample_dict["data.test_pipeline_2"], 7)

        sample_dict = pipe.reverse(sample_dict, "data.test_pipeline_2", "data.test_pipeline_2")
        self.assertEqual("data.test_pipeline" in sample_dict, False)
        self.assertEqual("data.test_pipeline_2" in sample_dict, False)

    def test_none(self):
        """
        Test pipeline with an op returning None
        """
        pipeline_seq = [
            (OpSetForTest(), dict(key="data.test_pipeline", val=5)),
            (OpNoneForTest(), dict()),
            (OpSetForTest(), dict(key="data.test_pipeline_2", val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)
        sample_dict = NDict({})
        sample_dict = pipe(sample_dict)
        self.assertIsNone(sample_dict)

    def test_split(self):
        """
        Test pipeline with an op splitting samples to multiple samples
        """
        pipeline_seq = [
            (OpSetForTest(), dict(key="data.test_pipeline", val=5)),
            (OpSplitForTest(), dict()),
            (OpSetForTest(), dict(key="data.test_pipeline_2", val=7)),
        ]
        pipe = PipelineDefault("test", pipeline_seq)
        sample_dict = NDict({"data": {"sample_id": 0}})
        sample_dict = pipe(sample_dict)
        self.assertTrue(isinstance(sample_dict, list))
        self.assertEqual(len(sample_dict), 10)
        expected_samples = [(0, i) for i in range(10)]
        samples = [sample["data.sample_id"] for sample in sample_dict]
        self.assertListEqual(expected_samples, samples)

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
