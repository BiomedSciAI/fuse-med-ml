from typing import Optional, Sequence, Tuple
from fuse.data import OpBase
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.split import dataset_balanced_division_to_folds

from fuseimg.datasets.isic import ISIC

from torch.utils.data.dataloader import DataLoader


def isic_2019_dataloaders(
    data_path: str,
    cache_path: str,
    reset_cache: bool = False,
    reset_split_file: bool = False,
    append_dyn_pipeline: Optional[Sequence[Tuple[OpBase, dict]]] = None,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train_dataloader and validation_dataloadr with specific parameters for image_clinical_multimodality tutorial
    :param data_path: path to download the data to
    :param cache_path: path to store the cached files
    :param reset_cache: restart cache or not
    :param reset_split_file: reuse previous split or create a new one.
    :param append_dyn_pipeline: steps to append at the end of the dynamic pipeline (doesn't require recaching)
    :param sample_ids: list of sample_ids tp include - otherwise will consider all the sample_ids
    :return: train_dataloader and validation_dataloader
    """
    # internal arguments used for this example
    n_folds = 3
    train_folds = [0, 1]
    validation_folds = [2]
    batch_size = 8
    num_workers = 4
    data_split_filename = "data_split.pkl"
    # split to folds randomly
    all_dataset = ISIC.dataset(
        data_path,
        cache_path,
        reset_cache=reset_cache,
        append_dyn_pipeline=append_dyn_pipeline,
        num_workers=num_workers,
        samples_ids=sample_ids,
    )

    folds = dataset_balanced_division_to_folds(
        dataset=all_dataset,
        output_split_filename=data_split_filename,
        keys_to_balance=["data.label"],
        nfolds=n_folds,
        reset_split=reset_split_file,
    )

    train_sample_ids = []
    for fold in train_folds:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in validation_folds:
        validation_sample_ids += folds[fold]

    train_dataset = ISIC.dataset(
        data_path, cache_path, samples_ids=train_sample_ids, append_dyn_pipeline=append_dyn_pipeline, train=True
    )

    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset, balanced_class_name="data.label", num_balanced_classes=8, batch_size=batch_size
    )
    print("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_sampler=sampler, collate_fn=CollateDefault(), num_workers=num_workers
    )

    # dataset
    validation_dataset = ISIC.dataset(
        data_path, cache_path, samples_ids=validation_sample_ids, append_dyn_pipeline=append_dyn_pipeline, train=False
    )

    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, collate_fn=CollateDefault(), num_workers=num_workers
    )

    return train_dataloader, validation_dataloader


SEX_INDEX = {"male": 0, "female": 1, "N/A": 2}
ANATOM_SITE_INDEX = {
    "anterior torso": 0,
    "upper extremity": 1,
    "posterior torso": 2,
    "lower extremity": 3,
    "lateral torso": 4,
    "head/neck": 5,
    "palms/soles": 6,
    "oral/genital": 7,
    "N/A": 8,
}
