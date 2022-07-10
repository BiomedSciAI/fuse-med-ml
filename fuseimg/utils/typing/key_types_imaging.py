from enum import Enum
from fuse.data.key_types import DataTypeBasic, TypeDetectorPatternsBased


class DataTypeImaging(Enum):
    """
    Possible data types stored in sample_dict.
    Using Patterns - the type will be inferred from the key name
    """

    IMAGE = "image"  # Image
    SEG = "seg"  # Segmentation Map
    BBOX = "bboxes"  # Bounding Box
    CTR = "contours"  # Contour


PATTERNS_DICT_IMAGING = {
    r".*img$": DataTypeImaging.IMAGE,
    r".*seg$": DataTypeImaging.SEG,
    r".*bbox$": DataTypeImaging.BBOX,
    r".*ctr$": DataTypeImaging.CTR,
    r".*$": DataTypeBasic.UNKNOWN,
}

type_detector_imaging = TypeDetectorPatternsBased(PATTERNS_DICT_IMAGING)
