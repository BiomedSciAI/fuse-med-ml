from fuse.data.ops.ops_common import OpApplyTypes
from fuse.data.key_types_for_testing import type_detector_for_testing
from functools import partial

OpApplyTypesImaging = partial(
    OpApplyTypes,
    type_detector=type_detector_for_testing,
)
