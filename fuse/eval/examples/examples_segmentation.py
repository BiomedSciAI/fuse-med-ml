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

import os
import pathlib
from typing import Any, Dict
from collections import OrderedDict
from pycocotools.coco import COCO
import numpy as np
import nibabel as nib
from fuse.utils import set_seed

from fuse.eval.metrics.segmentation.metrics_segmentation_common import (
    MetricDice,
    MetricIouJaccard,
    MetricOverlap,
    Metric2DHausdorff,
    MetricPixelAccuracy,
)
from fuse.eval.metrics.segmentation.metrics_instance_segmentation_common import MetricInstanceIouJaccard
from fuse.eval.metrics.detection.metrics_detection_common import MetricDetectionPrecision, MetricDetectionRecall
from fuse.eval.evaluator import EvaluatorDefault


def example_seg_0() -> Dict[str, Any]:
    """
    Simple evaluation example for semantic segmentation on 3 separate binary segmentations
    Inputs are 4 pairs of segmentation files: one including predictions and one targets
    new classes mapping is defined based on the the basic pixel segmentation values
    """
    LABEL_MAPPING = {
        "label_1": (1, 2, 3),
        "label_2": (2, 3),
        "label_3": (2,),
    }

    # pre collect function to change the format
    def pre_collect_process_label_1(sample_dict: dict) -> dict:
        # convert scores to numpy array
        mask = np.zeros(sample_dict["pred.array"].shape, dtype=bool)
        mask_label = np.zeros(sample_dict["pred.array"].shape, dtype=bool)
        for l in LABEL_MAPPING["label_1"]:
            mask[sample_dict["pred.array"] == l] = True
            mask_label[sample_dict["label.array"] == l] = True
        sample_dict["pred.array_label_1"] = mask
        sample_dict["label.array_label_1"] = mask_label
        return sample_dict

    def pre_collect_process_label_3(sample_dict: dict) -> dict:
        # convert scores to numpy array
        mask_pred = np.zeros(sample_dict["pred.array"].shape, dtype=bool)
        mask_label = np.zeros(sample_dict["pred.array"].shape, dtype=bool)
        for l in LABEL_MAPPING["label_3"]:
            mask_pred[sample_dict["pred.array"] == l] = True
            mask_label[sample_dict["label.array"] == l] = True
        sample_dict["pred.array_label_3"] = mask_pred
        sample_dict["label.array_label_3"] = mask_label
        return sample_dict

    def pre_collect_process_label_2(sample_dict: dict) -> dict:
        # convert scores to numpy array
        mask = np.zeros(sample_dict["pred.array"].shape, dtype=bool)
        mask_label = np.zeros(sample_dict["pred.array"].shape, dtype=bool)
        for l in LABEL_MAPPING["label_2"]:
            mask[sample_dict["pred.array"] == l] = True
            mask_label[sample_dict["label.array"] == l] = True
        sample_dict["pred.array_label_2"] = mask
        sample_dict["label.array_label_2"] = mask_label
        return sample_dict

    # define iterator
    def data_iter():
        dir_path = pathlib.Path(__file__).parent.resolve()
        predicted_list = os.listdir(os.path.join(dir_path, "inputs/semantic_segmentation/predicted/"))
        labels_path = os.path.join(dir_path, "inputs/semantic_segmentation/labeled/")
        for predicted in predicted_list:
            id = os.path.basename(predicted).split(".")[0]
            label_path = os.path.join(labels_path, id, "seg.nii.gz")
            predicted_path = os.path.join(dir_path, "inputs/semantic_segmentation/predicted/", predicted)
            sample_dict = {}
            sample_dict["id"] = id
            sample_dict["pred.array"] = np.asanyarray(nib.load(predicted_path).dataobj)
            sample_dict["label.array"] = np.asanyarray(nib.load(label_path).dataobj)
            yield sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "dice_label_1",
                MetricDice(
                    pred="pred.array_label_1",
                    target="label.array_label_1",
                    pre_collect_process_func=pre_collect_process_label_1,
                ),
            ),
            (
                "dice_label_2",
                MetricDice(
                    pred="pred.array_label_2",
                    target="label.array_label_2",
                    pre_collect_process_func=pre_collect_process_label_2,
                ),
            ),
            (
                "dice_label_3",
                MetricDice(
                    pred="pred.array_label_3",
                    target="label.array_label_3",
                    pre_collect_process_func=pre_collect_process_label_3,
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data_iter(), batch_size=1, metrics=metrics)

    return results


def example_seg_1() -> Dict[str, Any]:
    """
    Simple evaluation example for binary segmentation on metric score such as dice,overlap,pixel_accuracy,iou_jacard
    Inputs are pair of pixel array : one including predictions and one targets
    """

    # define iterator
    def data_iter():
        # set seed
        set_seed(0)

        sample_dict = {}
        sample_dict["id"] = id
        sample_dict["pred.array"] = np.random.rand(10, 5, 5, 4) > 0.5
        sample_dict["label.array"] = np.random.rand(10, 5, 5, 4) > 0.5
        yield sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            ("dice", MetricDice(pred="pred.array", target="label.array")),
            ("overlap", MetricOverlap(pred="pred.array", target="label.array")),
            ("pixel_accuracy", MetricPixelAccuracy(pred="pred.array", target="label.array")),
            ("iou_jaccard", MetricIouJaccard(pred="pred.array", target="label.array")),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data_iter(), batch_size=60, metrics=metrics)

    return results


def example_seg_2() -> Dict[str, Any]:
    """
    Simple evaluation example for dice score for multiclass semantic segmentation
    Inputs are 4 pairs of segmentation files: one including predictions and one targets
    """
    # define iterator
    def data_iter():
        dir_path = pathlib.Path(__file__).parent.resolve()
        predicted_list = os.listdir(os.path.join(dir_path, "inputs/semantic_segmentation/predicted/"))
        labels_path = os.path.join(dir_path, "inputs/semantic_segmentation/labeled/")
        for predicted in predicted_list:
            id = os.path.basename(predicted).split(".")[0]
            label_path = os.path.join(labels_path, id, "seg.nii.gz")
            predicted_path = os.path.join(dir_path, "inputs/semantic_segmentation/predicted/", predicted)
            sample_dict = {}
            sample_dict["id"] = id
            sample_dict["pred.array"] = np.asanyarray(nib.load(predicted_path).dataobj)
            sample_dict["label.array"] = np.asanyarray(nib.load(label_path).dataobj)
            yield sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            ("dice", MetricDice(pred="pred.array", target="label.array")),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data_iter(), batch_size=1, metrics=metrics)

    return results


def example_seg_3() -> Dict[str, Any]:
    """
    Simple evaluation example for 2d hausdorff distance for segmentation and the support of pixel weight in dice,overlap,pixel_accuracu,iou_jaccard
    Inputs are pair of pixel arrays : one including predictions and one targets
    """

    # define iterator
    def data_iter():
        sample_dict = {}
        sample_dict["id"] = id
        sample_dict["pred.array"] = np.array([(1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
        sample_dict["label.array"] = np.array([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
        sample_dict["pixel_weight"] = {"1": np.array([(0.125, 0.125), (0.125, 0.125), (0.125, 0.125), (0.125, 0.125)])}
        yield sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            ("hausdorff", Metric2DHausdorff(pred="pred.array", target="label.array", class_weights={"1": 1.0})),
            (
                "dice",
                MetricDice(
                    pred="pred.array", target="label.array", pixel_weight="pixel_weight", class_weights={"1": 1.0}
                ),
            ),
            (
                "overlap",
                MetricOverlap(
                    pred="pred.array", target="label.array", pixel_weight="pixel_weight", class_weights={"1": 1.0}
                ),
            ),
            (
                "pixel_accuracy",
                MetricPixelAccuracy(
                    pred="pred.array", target="label.array", pixel_weight="pixel_weight", class_weights={"1": 1.0}
                ),
            ),
            (
                "iou_jaccard",
                MetricIouJaccard(
                    pred="pred.array", target="label.array", pixel_weight="pixel_weight", class_weights={"1": 1.0}
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data_iter(), batch_size=60, metrics=metrics)

    return results


def example_seg_4() -> Dict[str, Any]:
    """
    Simple evaluation example for instance segmentation and detection based on COCO dataset
    Inputs are pair of pixel arrays : one including predictions and one targets
    """

    # define iterator
    def data_iter():
        dir_path = pathlib.Path(__file__).parent.resolve()
        annotation_path = os.path.join(dir_path, "inputs/detection/example_coco_new.json")
        cocoGt = COCO(annotation_path)
        # initialize COCO detections api
        resFile = os.path.join(dir_path, "inputs/detection/instances_val2014_fakesegm100_results.json")
        coco = cocoGt.loadRes(resFile)
        catNms = ["person", "car"]
        segtypes = ["bbox", "polygon"]
        map_field = {"polygon": "segmentation", "bbox": "bbox"}
        catIds = coco.getCatIds(catNms)
        imgIds = coco.getImgIds(catIds=catIds)
        for img_id in imgIds:
            img = coco.loadImgs(ids=[img_id])[0]
            sample_dict = {}
            sample_dict["id"] = id
            sample_dict["height"] = img["height"]
            sample_dict["width"] = img["width"]
            for index, catID in enumerate(catIds):
                pred_annIds = coco.getAnnIds(imgIds=img_id, catIds=[catID], iscrowd=None)
                target_annIds = cocoGt.getAnnIds(imgIds=img_id, catIds=[str(catID)], iscrowd=None)
                for type in segtypes:
                    pred_annotations = [seg[map_field[type]] for seg in coco.loadAnns(pred_annIds)]
                    target_annotations = [seg[map_field[type]] for seg in cocoGt.loadAnns(target_annIds)]
                    sample_dict[f"pred.array_{type}_{catNms[index]}"] = pred_annotations
                    sample_dict[f"label.array_{type}_{catNms[index]}"] = target_annotations
            yield sample_dict

    # list of metrics
    metrics = OrderedDict(
        [
            (
                "iou_bbox_person",
                MetricInstanceIouJaccard(
                    pred="pred.array_bbox_person",
                    target="label.array_bbox_person",
                    segmentation_pred_type="bbox",
                    segmentation_target_type="bbox",
                    height="height",
                    width="width",
                ),
            ),
            (
                "iou_bbox_car",
                MetricInstanceIouJaccard(
                    pred="pred.array_bbox_car",
                    target="label.array_bbox_car",
                    segmentation_pred_type="bbox",
                    segmentation_target_type="bbox",
                    height="height",
                    width="width",
                ),
            ),
            (
                "recall_bbox_person",
                MetricDetectionRecall(
                    pred="pred.array_bbox_person",
                    target="label.array_bbox_person",
                    segmentation_pred_type="bbox",
                    segmentation_target_type="bbox",
                    height="height",
                    width="width",
                ),
            ),
            (
                "recall_bbox_car",
                MetricDetectionRecall(
                    pred="pred.array_bbox_car",
                    target="label.array_bbox_car",
                    segmentation_pred_type="bbox",
                    segmentation_target_type="bbox",
                    height="height",
                    width="width",
                ),
            ),
            (
                "precision_bbox_person",
                MetricDetectionPrecision(
                    pred="pred.array_bbox_person",
                    target="label.array_bbox_person",
                    segmentation_pred_type="bbox",
                    segmentation_target_type="bbox",
                    height="height",
                    width="width",
                ),
            ),
            (
                "precision_bbox_car",
                MetricDetectionPrecision(
                    pred="pred.array_bbox_car",
                    target="label.array_bbox_car",
                    segmentation_pred_type="bbox",
                    segmentation_target_type="bbox",
                    height="height",
                    width="width",
                ),
            ),
            (
                "iou_polygon_person",
                MetricInstanceIouJaccard(
                    pred="pred.array_polygon_person",
                    target="label.array_polygon_person",
                    segmentation_pred_type="compressed_RLE",
                    segmentation_target_type="polygon",
                    height="height",
                    width="width",
                ),
            ),
            (
                "iou_polygon_car",
                MetricInstanceIouJaccard(
                    pred="pred.array_polygon_car",
                    target="label.array_polygon_car",
                    segmentation_pred_type="compressed_RLE",
                    segmentation_target_type="polygon",
                    height="height",
                    width="width",
                ),
            ),
            (
                "recall_polygon_person",
                MetricDetectionRecall(
                    pred="pred.array_polygon_person",
                    target="label.array_polygon_person",
                    segmentation_pred_type="compressed_RLE",
                    segmentation_target_type="polygon",
                    height="height",
                    width="width",
                ),
            ),
            (
                "recall_polygon_car",
                MetricDetectionRecall(
                    pred="pred.array_polygon_car",
                    target="label.array_polygon_car",
                    segmentation_pred_type="compressed_RLE",
                    segmentation_target_type="polygon",
                    height="height",
                    width="width",
                ),
            ),
            (
                "precision_polygon_person",
                MetricDetectionPrecision(
                    pred="pred.array_polygon_person",
                    target="label.array_polygon_person",
                    segmentation_pred_type="compressed_RLE",
                    segmentation_target_type="polygon",
                    height="height",
                    width="width",
                ),
            ),
            (
                "precision_polygon_car",
                MetricDetectionPrecision(
                    pred="pred.array_polygon_car",
                    target="label.array_polygon_car",
                    segmentation_pred_type="compressed_RLE",
                    segmentation_target_type="polygon",
                    height="height",
                    width="width",
                ),
            ),
        ]
    )

    evaluator = EvaluatorDefault()
    results = evaluator.eval(ids=None, data=data_iter(), batch_size=60, metrics=metrics)

    return results
