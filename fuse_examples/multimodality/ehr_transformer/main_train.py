from typing import Any, Optional, Tuple
import hydra
from omegaconf import DictConfig
from copy import deepcopy

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.models.heads import Head1D
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC
from fuse.dl.losses import LossDefault
from fuse.utils import NDict

from fuse_examples.multimodality.ehr_transformer.model import Embed, TransformerEncoder, Bert, BertConfig
from fuse_examples.multimodality.ehr_transformer.dataset import PhysioNetCinC


def filter_gender_label_unknown(batch_dict: NDict) -> NDict:
    """
    Ignore unlabeled (gender) when computing gender classification loss
    """
    # filter out samples
    keep_indices = batch_dict["Gender"] != -1
    batch_dict = batch_dict.indices(
        keep_indices
    )  # this will keep only a subset of elements per each key in the dictionary (where applicable)
    return batch_dict


def data(
    dataset_cfg: dict, target_key: str, batch_size: int, data_loader_train: dict, data_loader_valid: dict
) -> Tuple[Any, DataLoader, DataLoader]:
    """
    return token to index mapper and train and validation dataloaders for MIMICC II
    :param dataset_cfg: PhysioNetCinC.dataset arguments
    :param target_key: will be used to balance the training dataset
    :param data_loader_train: arguments for train dataloader
    :param data_loader_train: arguments for validation dataloader
    """

    token2idx, ds_train, ds_valid, _ = PhysioNetCinC.dataset(**dataset_cfg)

    dl_train = DataLoader(
        ds_train,
        collate_fn=CollateDefault(keep_keys=["data.sample_id", "Target", "Indexes", "Gender", "NextVisitLabels"]),
        batch_sampler=BatchSamplerDefault(
            ds_train,
            balanced_class_name=target_key,
            batch_size=batch_size,
            mode="approx",
            num_batches=500,
        ),
        **data_loader_train,
    )
    dl_valid = DataLoader(
        ds_valid,
        collate_fn=CollateDefault(keep_keys=["data.sample_id", "Target", "Indexes", "Gender", "NextVisitLabels"]),
        **data_loader_valid,
    )

    return token2idx, dl_train, dl_valid


def model(
    embed: dict,
    classifier_head: dict,
    z_dim: int,
    encoder_type: str,
    transformer_encoder: dict,
    bert_config_kwargs: dict,
    vocab_size: int,
    aux_gender_classification: bool,
    classifier_gender_head: dict,
    aux_next_vis_classification: bool,
    classifier_next_vis_head: dict,
) -> torch.nn.Module:

    """
    Create transformer based model with 3 classification heads
    :param embed: arguments for Embed constructor
    :param classifier_head: arguments for Head1D for main classification task
    :param z_dim: the output size of the transformer model (cls token)
    :param encoder_type: either "transformer" or "bert"
    :param transformer_encoder: arguments for TransformerEncoder - used when encoder_type is "transformer"
    :param bert_config_kwargs: arguments for BertConfig - used with encoder_type is "bert"
    :param vocab_size: vocabulary size
    :param  aux_gender_classification: enable gender auxiliary classification head
    :param classifier_gender_head: arguments for Head1D for gender classification task
    :param  aux_next_vis_classification: enable next visit auxiliary classification head
    :param classifier_next_vis_head: arguments for Head1D for next visit classification task
    """
    if encoder_type == "transformer":
        encoder_model = ModelWrapSeqToDict(
            model=TransformerEncoder(**transformer_encoder),
            model_inputs=["model.embedding"],
            model_outputs=["model.z", None],
        )
    elif encoder_type == "bert":
        bert_config = BertConfig(vocab_size_or_config_json_file=vocab_size, **bert_config_kwargs)

        encoder_model = ModelWrapSeqToDict(
            model=Bert(config=bert_config),
            model_inputs=["model.embedding"],
            model_outputs=["model.z"],
        )
    else:
        raise Exception(f"Error: unknown encoder_type {encoder_type}")

    models_sequence = [
        Embed(key_in="Indexes", key_out="model.embedding", n_vocab=vocab_size, **embed),
        encoder_model,
        Head1D(head_name="cls", conv_inputs=[("model.z", z_dim)], **classifier_head),
    ]

    # append auxiliary head for gender
    if aux_gender_classification:
        models_sequence.append(Head1D(head_name="gender", conv_inputs=[("model.z", z_dim)], **classifier_gender_head))

    if aux_next_vis_classification:
        models_sequence.append(
            Head1D(
                head_name="next_vis",
                conv_inputs=[("model.z", z_dim)],
                num_outputs=vocab_size,
                **classifier_next_vis_head,
            )
        )

    model = torch.nn.Sequential(*models_sequence)
    return model


def train(
    model: torch.nn.Module,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    model_dir: str,
    opt: callable,
    trainer_kwargs: dict,
    target_key: str,
    target_loss_weight: float,
    aux_gender_classification: bool,
    gender_loss_weight: float,
    aux_next_vis_classification: bool,
    next_vis_loss_weight: float,
    lr_scheduler: callable = None,
    track_clearml: Optional[dict] = None,
) -> None:
    """
    Run training process
    :param model: the model to train
    :param dl_train: train set dataloader
    :param dl_valid: validation set dataloader
    :param model_dir: path to store the training process outputs
    :param opt: callable that give model parameters will return torch optimizer
    :param trainer_kwargs: parameters to pl.Trainer
    :param target_key: key that points to main task labels
    :param target_loss_weight: weight for main task loss
    :param aux_gender_classification: enable gender auxiliary classification head
    :param gender_loss_weight: weight for gender classification task loss
    :param aux_next_vis_classification: enable next visit auxiliary classification head
    :param next_vis_loss_weight: weight for next visit classification task loss
    :param lr_scheduler: callable that given optimizer returns torch lr scheduler
    :param track_clearml: optional - to track with clearml provide arguments to start_clearml_logger()
    """

    if track_clearml is not None:
        from fuse.dl.lightning.pl_funcs import start_clearml_logger

        start_clearml_logger(**track_clearml)

    #  Loss
    losses = {
        "ce": LossDefault(
            pred="model.logits.cls", target=target_key, callable=F.cross_entropy, weight=target_loss_weight
        ),
    }

    # Metrics
    train_metrics = {
        "auc": MetricAUCROC(pred="model.output.cls", target=target_key),
    }

    # auxiliary gender loss and metric
    if aux_gender_classification:
        losses["gender_ce"] = LossDefault(
            pred="model.logits.gender",
            target="Gender",
            callable=F.cross_entropy,
            preprocess_func=filter_gender_label_unknown,
            weight=gender_loss_weight,
        )

        train_metrics["gender_auc"] = MetricAUCROC(
            pred="model.output.gender", target="Gender", batch_pre_collect_process_func=filter_gender_label_unknown
        )

    # auxiliary gender loss and metric
    if aux_next_vis_classification:
        losses["next_vis_ce"] = LossDefault(
            pred="model.logits.next_vis",
            target="NextVisitLabels",
            callable=torch.nn.BCEWithLogitsLoss(),  # multi binary labels loss
            weight=next_vis_loss_weight,
        )

    validation_metrics = deepcopy(train_metrics)
    # either a dict with arguments to pass to ModelCheckpoint or list dicts for multiple ModelCheckpoint callbacks (to monitor and save checkpoints for more then one metric).
    best_epoch_source = dict(
        monitor="validation.metrics.auc",
        mode="max",
    )

    # optimizer and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizer = opt(params=model.parameters())
    optimizers_and_lr_schs = dict(optimizer=optimizer)

    if lr_scheduler is not None:
        optimizers_and_lr_schs["lr_scheduler"] = lr_scheduler(optimizer)
        if isinstance(optimizers_and_lr_schs["lr_scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau):
            optimizers_and_lr_schs["monitor"] = "validation.losses.total_loss"

    #  Train

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

    # create lightning trainer.
    pl_trainer = pl.Trainer(**trainer_kwargs)

    # train
    pl_trainer.fit(pl_module, dl_train, dl_valid, ckpt_path=None)


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)

    # get data
    token2idx, dl_train, dl_valid = data(**cfg.data)

    # create model
    nn_model = model(vocab_size=len(token2idx), **cfg.model)

    # train the model
    train(model=nn_model, dl_train=dl_train, dl_valid=dl_valid, **cfg.train)


if __name__ == "__main__":
    """
    See README for instructions
    """
    main()
