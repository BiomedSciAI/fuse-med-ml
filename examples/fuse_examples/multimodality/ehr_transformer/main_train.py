from dataset import PhysioNetCinC
from typing import Any, Optional, List, Tuple
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
from fuse.eval.metrics.metrics_common import Filter
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC
from fuse.dl.losses import LossDefault
from fuse.utils import NDict
from fuse.data import DatasetDefault


from examples.fuse_examples.multimodality.ehr_transformer.model import Embed, TransformerEncoder

def filter_gender_label_unknown_for_loss(batch_dict: NDict) -> NDict:
    # filter out samples
    keep_indices = batch_dict["Gender"] != -1
    batch_dict = batch_dict.indices(
        keep_indices
    )  # this will keep only a subset of elements per each key in the dictionary (where applicable)
    return batch_dict


def filter_gender_label_unknown_for_metric(sample_dict: NDict) -> NDict:
    # filter out samples
    sample_dict["filter"] = sample_dict["Gender"] == -1
    return sample_dict

def data(
    dataset_cfg: dict, target_key: str, batch_size: int, data_loader_train: dict, data_loader_valid: dict
) -> Tuple[DatasetDefault, DataLoader, DataLoader]:

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
        ds_valid, collate_fn=CollateDefault(keep_keys=["data.sample_id", "Target", "Indexes", "Gender", "NextVisitLabels"]), **data_loader_valid
    )

    return token2idx, ds_train, dl_train, dl_valid


def model(
    embed: dict,
    classifier_head: dict,
    z_dim: int,
    transformer_encoder: dict,
    vocab_size: int,
    aux_gender_classification: bool,
    classifier_gender_head: dict,
    aux_next_vis_classification: bool,
    classifier_next_vis_head: dict
):
    embed = Embed(key_in="Indexes", key_out="model.embedding", n_vocab=vocab_size, **embed)

    encoder_model = TransformerEncoder(**transformer_encoder)

    models_sequence = [
        embed,
        ModelWrapSeqToDict(
            model=encoder_model,
            model_inputs=["model.embedding"],
            model_outputs=["model.z", None],
        ),
        Head1D(head_name="cls", conv_inputs=[("model.z", z_dim)], **classifier_head),
    ]

    # append auxiliary head for gender 
    if aux_gender_classification:
        models_sequence.append(Head1D(head_name="gender", conv_inputs=[("model.z", z_dim)], **classifier_gender_head))

    if aux_next_vis_classification:
        models_sequence.append(Head1D(head_name="next_vis", conv_inputs=[("model.z", z_dim)], num_outputs=vocab_size, **classifier_next_vis_head))

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
):

    if track_clearml is not None:
        from fuse.dl.lightning.pl_funcs import start_clearml_logger
        start_clearml_logger(**track_clearml)

    #  Loss
    losses = {
        "ce": LossDefault(
            pred="model.logits.cls",
            target=target_key,
            callable=F.cross_entropy,
            weight=target_loss_weight
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
            preprocess_func=filter_gender_label_unknown_for_loss,
            weight=gender_loss_weight

        )

        train_metrics["gender_auc"] =  Filter(MetricAUCROC(pred="model.output.gender", target="Gender"), "filter", pre_collect_process_func=filter_gender_label_unknown_for_metric)
        

    # auxiliary gender loss and metric 
    if aux_next_vis_classification:
        losses["next_vis_ce"] = LossDefault(
            pred="model.logits.next_vis",
            target="NextVisitLabels",
            callable=torch.nn.BCEWithLogitsLoss(), # multi binary labels loss
            weight=next_vis_loss_weight
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
def main(cfg: DictConfig):
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)

    token2idx, ds_train, dl_train, dl_valid = data(**cfg.data)

    # model
    nn_model = model(vocab_size=len(token2idx), **cfg.model)

    train(model=nn_model, dl_train=dl_train, dl_valid=dl_valid, **cfg.train)


if __name__ == "__main__":
    # TODO: delete
    # export CINC_DATA_PKL="/dccstor/mm_hcls/datasets/PhysioNet_CINC_2012/data.pkl"
    # export CINC_DATA_PATH="/dccstor/mm_hcls/datasets/PhysioNet_CINC_2012/"
    main()
