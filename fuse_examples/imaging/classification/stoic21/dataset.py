from typing import Union, Sequence, Optional
from fuse.data import DatasetDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuseimg.datasets.stoic21 import STOIC21


def train_val_test_splits(paths: Optional[dict] = None, params: Optional[dict] = None) -> Sequence[Sequence]:
    # split to folds randomly - temp
    dataset_all = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], reset_cache=False)
    folds = dataset_balanced_division_to_folds(
        dataset=dataset_all,
        output_split_filename=paths["data_split_filename"],
        keys_to_balance=["data.gt.probSevere"],
        nfolds=params["train"]["data.num_folds"],
    )

    test_sample_ids = []
    for fold in params["train"]["data.validation_folds"]:
        test_sample_ids += folds[fold]
    splits = []
    for fold in params["train"]["data.train_folds"]:
        train_sample_ids = sum([folds[f] for f in params["train"]["data.train_folds"] if f != fold], [])
        val_sample_ids = folds[fold]
        splits.append([train_sample_ids, val_sample_ids, test_sample_ids])

    return splits


def create_dataset(
    train_val_sample_ids: Union[Sequence, None] = None, paths: Optional[dict] = None, params: Optional[dict] = None
) -> Sequence[DatasetDefault]:
    if train_val_sample_ids is None:
        splits = train_val_test_splits(paths, params)
        train_val_sample_ids = sum(splits[0][0:2], [])
        test_sample_ids = splits[0][2]
        train_val_dataset = STOIC21.dataset(
            paths["data_dir"], paths["cache_dir"], sample_ids=train_val_sample_ids, train=True
        )  # note all of this dataset will get a "training" pipeline (with augmentations etc')
        test_dataset = STOIC21.dataset(paths["data_dir"], paths["cache_dir"], sample_ids=test_sample_ids, train=False)
        return train_val_dataset, test_dataset
    else:
        train_dataset = STOIC21.dataset(
            paths["data_dir"], paths["cache_dir"], sample_ids=train_val_sample_ids[0], train=True
        )
        validation_dataset = STOIC21.dataset(
            paths["data_dir"], paths["cache_dir"], sample_ids=train_val_sample_ids[1], train=False
        )

    return train_dataset, validation_dataset
