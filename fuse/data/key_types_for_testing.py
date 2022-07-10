from enum import Enum
from fuse.data.key_types import DataTypeBasic, TypeDetectorPatternsBased


class DataTypeForTesting(Enum):
    """
    Possible data types stored in sample_dict.
    Using Patterns - the type will be inferred from the key name
    """

    # Default options for types
    IMAGE_FOR_TESTING = (0,)  # Image
    SEG_FOR_TESTING = (1,)  # Segmentation Map
    BBOX_FOR_TESTING = (2,)  # Bounding Box
    CTR_FOR_TESTING = (3,)  # Contour


PATTERNS_DICT_FOR_TESTING = {
    r".*img_for_testing$": DataTypeForTesting.IMAGE_FOR_TESTING,
    r".*seg_for_testing$": DataTypeForTesting.SEG_FOR_TESTING,
    r".*bbox_for_testing$": DataTypeForTesting.BBOX_FOR_TESTING,
    r".*ctr_for_testing$": DataTypeForTesting.CTR_FOR_TESTING,
    r".*$": DataTypeBasic.UNKNOWN,
}

type_detector_for_testing = TypeDetectorPatternsBased(PATTERNS_DICT_FOR_TESTING)
