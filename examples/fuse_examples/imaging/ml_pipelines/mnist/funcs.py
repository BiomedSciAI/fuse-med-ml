from typing import OrderedDict, Union
from fuseimg.datasets.mnist import MNIST
from typing import Sequence
from fuse.data import DatasetDefault
from examples.fuse_examples.imaging.classification.mnist.run_mnist import create_model

def create_dataset(cache_dir: str, test: bool=True, train_val_sample_ids: Union[Sequence, None]=None) -> Sequence[DatasetDefault]:
    train_val_dataset = MNIST.dataset(cache_dir, train=True)
    if test or train_val_sample_ids is None:
        test_dataset = MNIST.dataset(cache_dir, train=False)
        return train_val_dataset, test_dataset
    else:
        train_dataset = MNIST.dataset(cache_dir, train=True, sample_ids=train_val_sample_ids[0])
        validation_dataset = MNIST.dataset(cache_dir, train=True, sample_ids=train_val_sample_ids[1])
        return train_dataset, validation_dataset
