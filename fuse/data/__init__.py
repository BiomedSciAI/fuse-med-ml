import os
import pathlib

# version
with open(os.path.join(pathlib.Path(__file__).parent, "..", "..", "VERSION.txt")) as version_file:
    __version__ = version_file.read().strip()

# import shortcuts
from fuse.data.utils.sample import get_sample_id, set_sample_id, get_sample_id_key
from fuse.data.utils.sample import create_initial_sample, get_initial_sample_id, get_initial_sample_id_key, get_specific_sample_from_potentially_morphed
from fuse.data.ops.op_base import OpBase #DataTypeForTesting, 
from fuse.data.ops.ops_common import OpApplyPatterns, OpLambda, OpFunc, OpRepeat, OpKeepKeypaths
from fuse.data.ops.ops_aug_common import OpRandApply, OpSample, OpSampleAndRepeat
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToTensor, OpToNumpy
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.export import ExportDataset
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetBase, DatasetDefault
