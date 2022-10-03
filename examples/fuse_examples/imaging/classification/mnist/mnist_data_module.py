from typing import List
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from fuse.data.utils.collates import CollateDefault
from fuseimg.datasets.mnist import MNIST
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds


class MNISTDataModule(pl.LightningDataModule):
    """
    Example of a custom Lightning datamodule using FuseMedML tools (folds + batch_sampler)

    For reference please visit:
    https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningDataModule.html
    """

    def __init__(
        self,
        cache_dir: str,
        num_workers: int,
        batch_size: int,
        train_folds: List[int],
        validation_folds: List[int],
        split_filename: str,
    ):
        """
        :param cache_dir: path to cache directory
        :param num_workers: number of processes to pass to dataloader and batch_sampler
        :param batch_size: batch_sampler's batch size
        :param train_folds: which folds will be used for training (#total_folds = #train_folds + #validation_folds)
        :param validation_folds: which folds will be used for validation (#total_folds = #train_folds + #validation_folds)
        :param split_filename: path to `division_to_folds` file
        """
        super().__init__()
        self._cache_dir = cache_dir
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._train_ids = []
        self._validation_ids = []

        # divide into tain and validation folds
        folds = dataset_balanced_division_to_folds(
            dataset=MNIST.dataset(self._cache_dir, train=True),
            keys_to_balance=["data.label"],
            nfolds=len(train_folds + validation_folds),
            output_split_filename=split_filename,
            reset_split=False,
        )

        for fold in train_folds:
            self._train_ids += folds[fold]

        for fold in validation_folds:
            self._validation_ids += folds[fold]

    def setup(self, stage: str):
        """
        creates datasets by stage
        called on every process in DDP

        :param stage: trainer stage
        """
        # assign train/val datasets for use in dataloaders
        if stage == "fit":
            self._mnist_train = MNIST.dataset(self._cache_dir, train=True, sample_ids=self._train_ids)
            self._mnist_validation = MNIST.dataset(self._cache_dir, train=True, sample_ids=self._validation_ids)

        # assign prediction (infer) dataset for use in dataloader
        if stage == "predict":
            self._mnist_predict = MNIST.dataset(self._cache_dir, train=False)

    def train_dataloader(self):
        """
        returns train dataloader with class args
        """
        # Create a batch sampler for the dataloader
        batch_sampler = BatchSamplerDefault(
            dataset=self._mnist_train,
            balanced_class_name="data.label",
            num_balanced_classes=10,
            batch_size=self._batch_size,
            workers=self._num_workers,
            verbose=True,
        )

        # Create dataloader
        train_dl = DataLoader(
            dataset=self._mnist_train,
            batch_sampler=batch_sampler,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
        )

        return train_dl

    def val_dataloader(self):
        """
        returns validation dataloader with class args
        """
        # Create dataloader
        validation_dl = DataLoader(
            dataset=self._mnist_validation,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )

        return validation_dl

    def predict_dataloader(self):
        """
        returns validation dataloader with class args
        """
        # Create dataloader
        predict_dl = DataLoader(
            dataset=self._mnist_predict,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return predict_dl
