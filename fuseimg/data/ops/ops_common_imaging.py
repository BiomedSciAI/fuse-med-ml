from fuse.data.ops.ops_common import OpApplyTypes
from fuseimg.utils.typing.key_types_imaging import type_detector_imaging
from functools import partial

OpApplyTypesImaging = partial(
    OpApplyTypes,
    type_detector=type_detector_imaging,
)
