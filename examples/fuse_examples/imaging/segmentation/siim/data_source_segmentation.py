import pandas as pd
from glob import glob
import random
import pickle
from typing import Sequence, Hashable, Union, Optional, List, Dict
from pathlib import Path

# from fuse.data.data_source.data_source_base import FuseDataSourceBase
# from fuse.utils.utils_misc import autodetect_input_source


def filter_files(files, include=[], exclude=[]):
    for incl in include:
        files = [f for f in files if incl in f.name]
    for excl in exclude:
        files = [f for f in files if excl not in f.name]
    return sorted(files)


def ls(x, recursive=False, include=[], exclude=[]):
    if not recursive:
        out = list(x.iterdir())
    else:
        out = [o for o in x.glob('**/*')]
    out = filter_files(out, include=include, exclude=exclude)
    return out


# class FuseDataSourceSeg():
#     def __init__(self, 
def get_data_sample_ids(            
                 phase: str, # can be ['train', 'validation']
                 data_folder: Optional[str] = None,
                 partition_file: Optional[str] = None,
                 val_split: float = 0.2,
                 override_partition: bool = True,
                 data_shuffle: bool = True
                 ):
        """
        Create DataSource
        :param input_source:       path to images
        :param partition_file:     Optional, name of a pickle file when no validation set is available
                                   If train = True, train/val indices are dumped into the file,
                                   If train = False, train/val indices are loaded
        :param train:              specifies if we are in training phase
        :param val_split:          validation proportion in case of splitting
        :param override_partition: specifies if the given partition file is filled with new train/val splits
        """

        # Extract entities
        # ----------------
        if partition_file is not None:
            if phase == 'train':
                if override_partition:

                    # rle_df = pd.read_csv(data_source)

                    Path.ls = ls
                    files = Path(data_folder).ls(recursive=True, include=['.dcm'])

                    sample_descs = [str(fn) for fn in files]
                    # sample_descs = []
                    # for fn in files:
                    #     I = rle_df.ImageId == fn.stem
                    #     desc = {'name': fn.stem,
                    #             'dcm': str(fn),
                    #             'rle_encoding': rle_df.loc[I, ' EncodedPixels'].values}
                    #     sample_descs.append(desc)
                    
                    if len(sample_descs) == 0:
                        raise Exception('Error detecting input source in FuseDataSourceDefault')

                    if data_shuffle:
                        # random shuffle the file-list
                        random.shuffle(sample_descs)

                    # split to train-validation -
                    n_train = int(len(sample_descs) * (1-val_split))

                    train_samples = sample_descs[:n_train]
                    val_samples = sample_descs[n_train:]
                    splits = {'train': train_samples, 'val': val_samples}

                    with open(partition_file, "wb") as pickle_out:
                        pickle.dump(splits, pickle_out)
                    sample_descs = train_samples
                else:
                    # read from a previous train/test split to evaluate on the same partition
                    with open(partition_file, "rb") as splits:
                        repartition = pickle.load(splits)
                    sample_descs = repartition['train']
            elif phase == 'validation':
                with open(partition_file, "rb") as splits:
                    repartition = pickle.load(splits)
                sample_descs = repartition['val']
        else:
            rle_df = pd.read_csv(data_source)

            Path.ls = ls
            files = Path(data_folder).ls(recursive=True, include=['.dcm'])

            sample_descs = [str(fn) for fn in files]
            # sample_descs = []
            # for fn in files:
            #     I = rle_df.ImageId == fn.stem
            #     desc = {'name': rle_df.loc[I, 'ImageId'].values[0],
            #             'dcm': fn,
            #             'rle_encoding': rle_df.loc[I, ' EncodedPixels'].values}
            #     sample_descs.append(desc)

        return sample_descs

    # def get_samples_description(self):
    #     return self.samples

    # def summary(self) -> str:
    #     summary_str = ''
    #     summary_str += 'FuseDataSourceSeg - %d samples\n' % len(self.samples)
    #     return summary_str
