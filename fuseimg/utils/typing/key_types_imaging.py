from enum import Enum
from fuse.data.key_types import DataTypeBasic, TypeDetectorPatternsBased
from typing import *

class DataTypeImaging(Enum):
    """
    Possible data types stored in sample_dict.
    Using Patterns - the type will be inferred from the key name
    """
    IMAGE = "image"  # Image
    SEG = "seg"  # Segmentation Map in numpy format
    BBOX = "bboxes"  # Bounding Box in format [bbox_x, bbox_y, bbox_w, bbox_h]
    CTR = "contours"  # Contour in format [x1,y1,x2,y2...]
    CRLE ="compressed rle" # COCO format Compressed Run-Lenght encoding
    UCRLE ="uncompressed rle" # COCO format Run-Lenght encoding -https://en.wikipedia.org/wiki/Run-length_encoding 

PATTERNS_DICT_IMAGING = {
    r".*img$": DataTypeImaging.IMAGE,
    r".*seg$": DataTypeImaging.SEG,
    r".*bbox$": DataTypeImaging.BBOX,
    r".*ctr$": DataTypeImaging.CTR,
    r".*crle$": DataTypeImaging.CRLE,
    r".*ucrle$": DataTypeImaging.UCRLE,
    r".*$": DataTypeBasic.UNKNOWN
}

type_detector_imaging = TypeDetectorPatternsBased(PATTERNS_DICT_IMAGING)
