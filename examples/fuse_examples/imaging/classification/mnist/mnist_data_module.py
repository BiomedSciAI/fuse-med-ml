from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from fuse.data.utils.collates import CollateDefault

from fuseimg.datasets.mnist import MNIST
from fuse.data.utils.samplers import BatchSamplerDefault


class MNISTDataModule(pl.LightningDataModule):
    """
    TODO use batch sampler
    """

    def __init__(self, cache_dir: str, num_workers: int, batch_size: int):
        """ """
        super().__init__()
        self._cache_dir = cache_dir
        self._num_workers = num_workers
        self._batch_size = batch_size
        return

    def train_dataloader(self):
        """
        returns train dataloader with custom args
        """
        train_dataset = MNIST.dataset(self._cache_dir, train=True)

        batch_sampler = BatchSamplerDefault(
            dataset=train_dataset,
            balanced_class_name="data.label",
            num_balanced_classes=10,
            batch_size=self._batch_size,
            verbose=True,
        )

        train_dl = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
        )

        return train_dl

    def val_dataloader(self):
        """
        returns validation dataloader with custom args
        """
        validation_dataset = MNIST.dataset(self._cache_dir, train=False)

        validation_dl = DataLoader(
            dataset=validation_dataset,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return validation_dl

    def predict_dataloader(self):
        """
        NOTE: In MNIST example we use the same data for validation and evaluation. (TODO change (?)
        """
        return self.val_dataloader()


if __name__ == "__main__":
    from fuse_examples.imaging.classification.mnist.run_mnist import PATHS

    dm = MNISTDataModule(cache_dir=PATHS["cache_dir"], batch_size=20, num_workers=10)
