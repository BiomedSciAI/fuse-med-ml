from typing import Optional, Sequence, Union

from fuse.data import DatasetDefault
from fuseimg.datasets.mnist import MNIST


def create_dataset(
    train_val_sample_ids: Union[Sequence, None] = None,
    paths: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Sequence[DatasetDefault]:
    train_val_dataset = MNIST.dataset(paths["cache_dir"], train=True)
    if train_val_sample_ids is None:
        test_dataset = MNIST.dataset(paths["cache_dir"], train=False)
        return train_val_dataset, test_dataset
    else:
        train_dataset = MNIST.dataset(
            paths["cache_dir"], train=True, sample_ids=train_val_sample_ids[0]
        )
        validation_dataset = MNIST.dataset(
            paths["cache_dir"], train=True, sample_ids=train_val_sample_ids[1]
        )
        return train_dataset, validation_dataset
