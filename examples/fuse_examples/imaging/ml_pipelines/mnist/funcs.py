import copy
import pytorch_lightning as pl
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe
import logging
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch
from typing import OrderedDict, Union
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
import torch.optim as optim
import os
import torch.nn.functional as F
from fuse.eval.evaluator import EvaluatorDefault
from fuseimg.datasets.mnist import MNIST
from typing import Sequence, Optional
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

def run_train(
    dataset: DatasetDefault,
    sample_ids: Sequence,
    cv_index: int,
    test: bool = False,
    params: Optional[dict] = None,
    rep_index: int = 0,
    rand_gen: Optional[torch.Generator] = None,
) -> None:
    assert test is False
    # obtain train/val dataset subset:
    train_dataset = MNIST.dataset(params["paths"]["cache_dir"], train=True, sample_ids=sample_ids[0])
    validation_dataset = MNIST.dataset(params["paths"]["cache_dir"], train=True, sample_ids=sample_ids[1])

    model_dir = os.path.join(params["paths"]["model_dir"], "rep_" + str(rep_index), str(cv_index))

    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Fuse Train")

    # ==============================================================================
    # Data
    # ==============================================================================
    ## Train data
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=10,
        batch_size=params["data.batch_size"],
        balanced_class_weights=None,
    )
    print("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=params["data.train_num_workers"],
    )
    print("Data - trainset: Done")

    ## Validation data
    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=params["data.batch_size"],
        collate_fn=CollateDefault(),
        num_workers=params["data.validation_num_workers"],
    )
    print("Data - validation set: Done")

    # ====================================================================================
    # Model
    # ====================================================================================
    model = create_model()

    # ====================================================================================
    # Losses
    # ====================================================================================
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
        ),
    }

    # ====================================================================================
    # Metrics
    # ====================================================================================
    train_metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
        ]
    )

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.accuracy",
        mode="max",
    )

    # ====================================================================================
    # Training components
    # ====================================================================================
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params["opt.learning_rate"], weight_decay=params["opt.weight_decay"])

    # create learning scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # ====================================================================================
    # Train
    # ====================================================================================
    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=params["trainer.num_epochs"],
        accelerator=params["trainer.accelerator"],
        strategy=params["trainer.strategy"],
        devices=params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=params["trainer.ckpt_path"])
    print("Train: Done")


def run_infer(
    dataset: DatasetDefault,
    sample_ids: Sequence,
    cv_index: int,
    test: bool = False,
    params: Optional[dict] = None,
    rep_index: int = 0,
    rand_gen: Optional[torch.Generator] = None,
) -> None:
    # obtain train/val dataset subset:
    if sample_ids is None:
        validation_dataset = dataset
    else:
        validation_dataset = Subset(dataset, sample_ids[1])

    #### Logger
    model_dir = os.path.join(params["paths"]["model_dir"], "rep_" + str(rep_index), str(cv_index))
    if test:
        inference_dir = os.path.join(params["paths"]["test_dir"], "rep_" + str(rep_index), str(cv_index))
        infer_filename = params["test_infer_filename"]
    else:
        inference_dir = os.path.join(params["paths"]["inference_dir"], "rep_" + str(rep_index), str(cv_index))
        infer_filename = params["infer_filename"]
    checkpoint_filename = params["checkpoint_filename"]

    create_dir(inference_dir)
    infer_file = os.path.join(inference_dir, infer_filename)
    checkpoint_file = os.path.join(model_dir, checkpoint_filename)
    #### Logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)

    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    ## Data
    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset, collate_fn=CollateDefault(), batch_size=2, num_workers=2
    )

    # load pytorch lightning module
    model = create_model()
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=model_dir, model=model, map_location="cpu", strict=True
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.output.classification", "data.label"]
    )  # which keys to extract and dump into file

    print("Model: Done")
    # create a trainer instance
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        accelerator=params["trainer.accelerator"],
        devices=params["trainer.num_devices"],
        strategy=params["trainer.strategy"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, validation_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


def run_eval(
    dataset: DatasetDefault,
    sample_ids: Sequence,
    cv_index: int,
    test: bool = False,
    params: Optional[dict] = None,
    rep_index: int = 0,
    rand_gen: Optional[torch.Generator] = None,
    pred_key: str = "model.output.classification",
    label_key: str = "data.label",
) -> None:
    if test:
        inference_dir = os.path.join(params["paths"]["test_dir"], "rep_" + str(rep_index), str(cv_index))
        infer_filename = params["test_infer_filename"]
    else:
        inference_dir = os.path.join(params["paths"]["inference_dir"], "rep_" + str(rep_index), str(cv_index))
        infer_filename = params["infer_filename"]
    if cv_index == "ensemble":
        infer_filename = "ensemble_results.gz"
    fuse_logger_start(output_path=inference_dir, console_verbose_level=logging.INFO)
    print("Fuse Eval")

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred=pred_key)),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target=label_key)),
            (
                "roc",
                MetricROCCurve(
                    pred=pred_key,
                    target=label_key,
                    class_names=class_names,
                    output_filename=os.path.join(inference_dir, "roc_curve.png"),
                ),
            ),
            ("auc", MetricAUCROC(pred=pred_key, target=label_key, class_names=class_names)),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    _ = evaluator.eval(
        ids=None, data=os.path.join(inference_dir, infer_filename), metrics=metrics, output_dir=inference_dir
    )
