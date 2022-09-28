from typing import List
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from fuse.data.utils.collates import CollateDefault
from fuseimg.datasets.mnist import MNIST
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds


class MNISTDataModule(pl.LightningDataModule):
    """
    TODO
    """

    def __init__(
        self, cache_dir: str, num_workers: int, batch_size: int, train_folds: List[int], validation_folds: List[int], split_filename: str
    ):
        """ """
        super().__init__()
        self._cache_dir = cache_dir
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._train_ids = []
        self._validation_ids = []

        print(f"split_filename={split_filename}")

        folds = dataset_balanced_division_to_folds(
            dataset=MNIST.dataset(self._cache_dir, train=True),
            keys_to_balance=["data.label"],
            nfolds=len(train_folds) + len(validation_folds),
            output_split_filename=split_filename,
            reset_split = True,
        )

        print(folds)

        for fold in train_folds:
            self._train_ids += folds[fold]

        for fold in validation_folds:
            self._validation_ids += folds[fold]

    def setup(self, stage: str):

        if stage == "fit":
            self._mnist_train = MNIST.dataset(self._cache_dir, train=True, sample_ids=self._train_ids)
            self._mnist_validation = MNIST.dataset(self._cache_dir, train=True, sample_ids=self._validation_ids)

        if stage == "predict":
            self._mnist_predict = MNIST.dataset(self._cache_dir, train=False)

    def train_dataloader(self):
        """
        returns train dataloader with custom args
        """
        batch_sampler = BatchSamplerDefault(
            dataset=self._mnist_train,
            balanced_class_name="data.label",
            num_balanced_classes=10,
            batch_size=self._batch_size,
            verbose=True,
        )

        train_dl = DataLoader(
            dataset=self._mnist_train,
            batch_sampler=batch_sampler,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
        )

        return train_dl

    def val_dataloader(self):
        """
        returns validation dataloader with custom args
        """
        validation_dl = DataLoader(
            dataset=self._mnist_validation,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return validation_dl

    def predict_dataloader(self):
        """
        NOTE: In MNIST example we use the same data for validation and evaluation. (TODO change (?)
        """

        predict_dl = DataLoader(
            dataset=self._mnist_predict,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return predict_dl


if __name__ == "__main__":
    from fuse_examples.imaging.classification.mnist.run_mnist import PATHS

    dm = MNISTDataModule(
        cache_dir=PATHS["cache_dir"],
        batch_size=20,
        num_workers=10,
        train_folds=[1, 2, 3, 4],
        validation_folds=[5],
    )
