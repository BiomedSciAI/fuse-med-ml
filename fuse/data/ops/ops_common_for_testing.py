from functools import partial

from fuse.data.key_types_for_testing import type_detector_for_testing
from fuse.data.ops.ops_common import OpApplyTypes

OpApplyTypesImaging = partial(
    OpApplyTypes,
    type_detector=type_detector_for_testing,
)
