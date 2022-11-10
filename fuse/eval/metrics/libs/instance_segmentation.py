from typing import Dict, Sequence

import numpy as np

from pycocotools import mask as maskUtils


class MetricsInstanceSegmentaion:
    @staticmethod
    def convert_uncompressed_RLE_COCO_type(element: Dict, height: int, width: int) -> Dict:
        """
        converts uncompressed RLE to COCO default type ( compressed RLE)
        :param element:  input in uncompressed Run Length Encoding (RLE - https://en.wikipedia.org/wiki/Run-length_encoding)
                        saved in map object example :  {"size": [333, 500],"counts": [26454, 2, 651, 3, 13, 1]}
                        counts first element is how many bits are not the object, then how many bits are the object and so on
        :param height: original image height in pixels
        :param width: original image width in pixels
        :return  COCO default type ( compressed RLE)
        """
        p = maskUtils.frPyObjects(element, height, width)
        return p

    @staticmethod
    def convert_polygon_COCO_type(element: list, height: int, width: int) -> Dict:
        """
        converts polygon to COCO default type ( compressed RLE)
        :param element:  polygon - array of X,Y coordinates saves as X.Y , example: [[486.34, 239.01, 477.88, 244.78]]
        :param height: original image height in pixels
        :param width: original image width in pixels
        :return  COCO default type ( compressed RLE)
        """
        rles = maskUtils.frPyObjects(element, height, width)
        p = maskUtils.merge(rles)
        return p

    @staticmethod
    def convert_pixel_map_COCO_type(element: np.ndarray) -> Dict:
        """
        converts pixel map (np.ndarray) to COCO default type ( compressed RLE)
        :param element:  pixel map in np.ndarray with same shape as original image ( 0= not the object, 1= object)
        :return  COCO default type ( compressed RLE)
        """
        p = maskUtils.encode(np.asfortranarray(element).astype(np.uint8))
        return p

    @staticmethod
    def convert_to_COCO_type(input_list: Sequence[np.ndarray], height: int, width: int, segmentation_type: str) -> Dict:
        """
        converts all input list to COCO default type ( compressed RLE)
        :param input_list:  list of input in any COCO format
        :param height: original image height in pixels
        :param width: original image width in pixels
        :param segmentation_type: input format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :return  list of output in COCO default type ( compressed RLE)
        """
        if segmentation_type == "uncompressed_RLE":
            output_list = [
                MetricsInstanceSegmentaion.convert_uncompressed_RLE_COCO_type(element, height, width)
                for element in input_list
            ]
        elif segmentation_type == "polygon":
            output_list = [
                MetricsInstanceSegmentaion.convert_polygon_COCO_type(element, height, width) for element in input_list
            ]
        elif segmentation_type == "pixel_map":
            output_list = [MetricsInstanceSegmentaion.convert_pixel_map_COCO_type(element) for element in input_list]

        return output_list

    @staticmethod
    def iou_jaccard(
        pred: Sequence[np.ndarray],
        target: Sequence[np.ndarray],
        segmentation_pred_type: str,
        segmentation_target_type: str,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Compute iou (jaccard) score using pycocotools functions
        :param pred: sample prediction inputs - list of segmentations supported by COCO
        :param target: sample target inputs - list of segmentations supported by COCO
        :param segmentation_pred_type: input pred format - pixel_map , uncompressed_RLE , compressed_RLE , polygon , bbox
        :param segmentation_target_type: input target format - pixel_map , uncompressed_RLE ,polygon ( undergoes convertion to uncompressed_RLE ,see description above)
                                                               compressed_RLE , bbox - array in X.Y coordinates for example  [445.56, 205.16, 54.44, 71.55] ,
        :param height: height of the original image ( y axis)
        :param width: width of the original image ( x axis)
        :return matrix of iou computed between each element from pred and target
        """
        # convert input format to supported in pycocotools
        if segmentation_pred_type in ["pixel_map", "uncompressed_RLE", "polygon"]:
            pred = MetricsInstanceSegmentaion.convert_to_COCO_type(pred, height, width, segmentation_pred_type)
        if segmentation_target_type in ["pixel_map", "uncompressed_RLE", "polygon"]:
            target = MetricsInstanceSegmentaion.convert_to_COCO_type(target, height, width, segmentation_target_type)
        if segmentation_target_type == "uncompressed_RLE":
            iscrowd = list(np.ones(len(target)))
        else:
            iscrowd = list(np.zeros(len(target)))
        scores = maskUtils.iou(pred, target, iscrowd)
        return scores
