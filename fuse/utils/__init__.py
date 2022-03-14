from gzip import READ
from fuse.utils.ndict import NDict
from fuse.utils.collate import CollateToBatchList, uncollate
# align to future changes
from fuse.utils.utils_file import FuseUtilsFile
read_dataframe = FuseUtilsFile.read_dataframe
save_dataframe = FuseUtilsFile.save_dataframe
from fuse.utils.utils_misc import FuseUtilsMisc
set_seed = FuseUtilsMisc.set_seed

