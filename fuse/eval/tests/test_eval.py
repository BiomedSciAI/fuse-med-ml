"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

from distutils.log import warn
import unittest


from fuse.eval.examples.examples import (
    example_0,
    example_1,
    example_2,
    example_3,
    example_4,
    example_5,
    example_6,
    example_7,
    example_8,
    example_9,
    example_10,
    example_11,
    example_12,
    example_13,
    example_14,
)

from fuse.eval.examples.examples_segmentation import (
    example_seg_0,
    example_seg_1,
    example_seg_2,
    example_seg_3,
    example_seg_4,
)


class TestEval(unittest.TestCase):
    def test_eval_example_0(self):
        results = example_0()
        self.assertEqual(results["metrics.auc"], 0.845)

    def test_eval_example_1(self):
        results = example_1()
        self.assertEqual(results["metrics.auc"], 0.845)

    def test_eval_example_2(self):
        results = example_2()
        self.assertEqual(results["metrics.auc"], 0.845)
        self.assertAlmostEqual(results["metrics.auc_per_fold.std"], 0.0019, places=3)
        self.assertAlmostEqual(results["metrics.auc_per_fold.mean"], 0.839, places=3)
        self.assertAlmostEqual(results["metrics.auc_per_fold.median"], 0.839, places=3)
        self.assertAlmostEqual(results["metrics.auc_per_fold.0"], 0.841, places=3)
        self.assertAlmostEqual(results["metrics.auc_per_fold.1"], 0.837, places=3)

    def test_eval_example_3(self):
        results = example_3()
        self.assertAlmostEqual(results["metrics.auc_per_group.female"], 1.000, places=2)
        self.assertAlmostEqual(results["metrics.auc_per_group.male"], 0.666, places=2)

    def test_eval_example_4(self):
        results = example_4()
        self.assertEqual(results["metrics.auc.org"], 0.845)
        self.assertAlmostEqual(results["metrics.auc.mean"], 0.8456, places=3)
        self.assertAlmostEqual(results["metrics.auc.conf_lower"], 0.712, places=3)
        self.assertAlmostEqual(results["metrics.auc.conf_upper"], 0.954, places=3)
        self.assertAlmostEqual(results["metrics.auc.std"], 0.0622, places=3)

    def test_eval_example_5(self):
        results = example_5()
        self.assertAlmostEqual(
            results["metrics.compare_a_to_b.sensitivity.p_value"],
            1 - (0.33 + 0.67 * 0.33 + 0.67 * 0.67 * 0.33),
            places=2,
        )
        self.assertEqual(results["metrics.compare_a_to_b.sensitivity.count"], 10000)

    def test_eval_example_6(self):
        results = example_6()
        self.assertAlmostEqual(results["metrics.delongs_test.p_value.macro_avg"], 0.3173, places=4)
        self.assertAlmostEqual(results["metrics.delongs_test.z.macro_avg"], 1.000, places=3)

    def test_eval_example_7(self):
        results = example_7()
        self.assertAlmostEqual(results["metrics.delongs_test.p_value"], 0.09453, places=5)
        self.assertAlmostEqual(results["metrics.delongs_test.z"], 1.672, places=3)

    def test_eval_example_8(self):
        results = example_8()
        self.assertAlmostEqual(results["metrics.auc.macro_avg"], 0.808, places=3)
        self.assertAlmostEqual(results["metrics.auc.B"], 0.877, places=3)
        self.assertAlmostEqual(results["metrics.auc.VHR"], 0.723, places=3)

    def test_eval_example_9(self):
        try:
            import fuse.data
            import torchvision
        except ImportError:
            warn(" test_eval_example_8: requires fuse-med-ml-data and torchvision packages")
            return

        results = example_9()
        self.assertAlmostEqual(results["metrics.accuracy"], 0.1, places=2)

    def test_eval_example_10(self):
        results = example_10()
        self.assertAlmostEqual(results["metrics.mcnemars_test.statistic"], 1.0, places=1)
        self.assertAlmostEqual(results["metrics.mcnemars_test.p_value"], 1.0, places=5)

    def test_eval_example_11(self):
        results = example_11()
        self.assertEqual(results["metrics.accuracy"], 0.5)

    def test_eval_example_12(self):
        results = example_12()
        self.assertAlmostEqual(results["metrics.acc"], 0.615, places=2)

    def test_eval_example_14(self):
        results = example_14()
        self.assertAlmostEqual(results["metrics.cindex_per_group.male"], 0.66, places=2)
        self.assertAlmostEqual(results["metrics.cindex_per_group.female"], 1.0, places=2)

    def test_eval_example_seg_0(self):
        results = example_seg_0()
        self.assertAlmostEqual(results["metrics.dice_label_1.1"], 0.8740, places=3)
        self.assertAlmostEqual(results["metrics.dice_label_2.1"], 0.7163, places=3)
        self.assertAlmostEqual(results["metrics.dice_label_3.1"], 0.7184, places=3)

    def test_eval_example_seg_1(self):
        results = example_seg_1()
        self.assertAlmostEqual(results["metrics.dice.1"], 0.49900, places=3)
        self.assertAlmostEqual(results["metrics.overlap.1"], 0.51759, places=3)
        self.assertAlmostEqual(results["metrics.pixel_accuracy.1"], 0.48167, places=3)
        self.assertAlmostEqual(results["metrics.iou_jaccard.1"], 0.3324, places=3)

    def test_eval_example_seg_2(self):
        results = example_seg_2()
        self.assertAlmostEqual(results["metrics.dice.1"], 0.9726, places=3)
        self.assertAlmostEqual(results["metrics.dice.2"], 0.7184, places=3)
        self.assertAlmostEqual(results["metrics.dice.average"], 0.8455, places=3)

    def test_eval_example_seg_3(self):
        results = example_seg_3()
        self.assertEqual(results["metrics.hausdorff.average"], 1.0)
        self.assertAlmostEqual(results["metrics.dice.average"], 0.6667, places=3)
        self.assertAlmostEqual(results["metrics.overlap.average"], 1.0, places=3)
        self.assertAlmostEqual(results["metrics.pixel_accuracy.average"], 0.5, places=3)
        self.assertAlmostEqual(results["metrics.iou_jaccard.average"], 0.5, places=3)

    def test_eval_example_seg_4(self):
        results = example_seg_4()
        self.assertAlmostEqual(results["metrics.iou_bbox_person"], 0.81210, places=3)
        self.assertAlmostEqual(results["metrics.iou_bbox_car"], 0.895, places=3)
        self.assertAlmostEqual(results["metrics.recall_bbox_person"], 0.6875, places=3)
        self.assertAlmostEqual(results["metrics.recall_bbox_car"], 0.9318, places=3)
        self.assertAlmostEqual(results["metrics.precision_bbox_person"], 1.0, places=3)
        self.assertAlmostEqual(results["metrics.iou_polygon_car"], 0.8776, places=3)
        self.assertAlmostEqual(results["metrics.recall_polygon_person"], 0.60416, places=3)
        self.assertAlmostEqual(results["metrics.recall_polygon_car"], 0.8068, places=3)
        self.assertAlmostEqual(results["metrics.precision_polygon_person"], 0.875, places=3)
        self.assertAlmostEqual(results["metrics.precision_polygon_car"], 0.875, places=3)

    def test_eval_example_13(self):
        results = example_13()
        self.assertAlmostEqual(results["metrics.reliability"]["avg_accuracy"], 0.566, places=2)
        self.assertAlmostEqual(results["metrics.reliability"]["avg_confidence"], 0.441, places=2)
        self.assertAlmostEqual(results["metrics.ece"]["ece"], 0.126, places=2)
        self.assertAlmostEqual(results["metrics.ece"]["mce"], 0.800, places=2)
        self.assertAlmostEqual(results["metrics.find_temperature"], 0.8076, places=3)
        self.assertAlmostEqual(results["metrics.ece_calibrated"]["ece"], 0.123, places=2)
        self.assertAlmostEqual(results["metrics.ece_calibrated"]["mce"], 0.300, places=2)
        self.assertAlmostEqual(results["metrics.reliability_calibrated"]["avg_accuracy"], 0.566, places=2)
        self.assertAlmostEqual(results["metrics.reliability_calibrated"]["avg_confidence"], 0.485, places=2)

    def test_eval_example_14(self):
        results = example_14()
        self.assertGreater(results["metrics.accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
