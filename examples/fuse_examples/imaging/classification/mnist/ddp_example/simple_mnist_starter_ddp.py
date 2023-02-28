from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist

# fuse stuff
import copy
from typing import OrderedDict
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.lightning.pl_module import LightningModuleDefault

from fuseimg.datasets.mnist import MNIST as fuse_MNIST
from examples.fuse_examples.imaging.classification.mnist import lenet

from fuse.dl.lightning.pl_funcs import start_clearml_logger

import os
import subprocess

import argparse

import lovely_tensors as lt

lt.monkey_patch()


## Paths and Hyperparameters ############################################
ROOT = "/dccstor/mm_hcls/shatz/mnist_ddp_test/"  # TODO: fill path here
MODEL_DIR = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": MODEL_DIR,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
}
TRAIN_PARAMS = {
    # "data.batch_size": 64,
    "data.num_workers": 16,
    "trainer.num_epochs": 30,
    # "trainer.num_nodes": 1,
    # "trainer.num_devices": 1,
    "trainer.accelerator": "gpu",
    # "trainer.strategy": "ddp",
    "trainer.ckpt_path": None,
    # "opt.lr": 4e-4,
    "opt.weight_decay": 0.001,
}


def check_nvidia_smi():
    try:
        print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print("ERRORL: Had a problem running nvidia-smi !")
        print(e)


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


# class MetricLogBatchSize():


class LitClassifier(LightningModuleDefault):
    def __init__(self, model_dir, backbone, learning_rate=1e-3):
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        model = ModelWrapSeqToDict(
            model=backbone,
            model_inputs=["data.image"],
            post_forward_processing_function=lambda logits: (logits, F.softmax(logits, dim=1)),
            model_outputs=["model.logits.classification", "model.output.classification"],
        )
        losses = {
            "cls_loss": LossDefault(
                pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
            )
        }
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        train_metrics = OrderedDict(
            [
                ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
                ("auc", MetricAUCROC(pred="model.output.classification", target="data.label", class_names=class_names)),
                ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
            ]
        )
        validation_metrics = copy.deepcopy(train_metrics)  # same as train metrics
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        optimizers_and_lr_schs = dict(optimizer=optimizer)
        super().__init__(
            model_dir=None,
            model=model,
            losses=losses,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            optimizers_and_lr_schs=optimizers_and_lr_schs,
        )


def run_train(paths, train_params):
    pl.seed_everything(1234)
    num_nodes = train_params["trainer.num_nodes"]
    num_gpus = train_params["trainer.num_devices"]
    bs = train_params["data.batch_size"]
    lr = train_params["opt.lr"]

    # log clearml only on rank 0
    start_clearml_logger(
        project_name="shatz_root/shatz_ddp_mnist_7",
        task_name=f"TEST num_nodes={num_nodes} num_gpus={num_gpus} bs={bs} lr={lr}",
    )

    mnist_train = fuse_MNIST.dataset(paths["cache_dir"], train=True)
    mnist_val = fuse_MNIST.dataset(paths["cache_dir"], train=False)

    train_loader = DataLoader(
        mnist_train,
        batch_size=train_params["data.batch_size"],
        num_workers=train_params["data.num_workers"],
        collate_fn=CollateDefault(),
    )
    val_loader = DataLoader(
        mnist_val,
        batch_size=train_params["data.batch_size"],
        num_workers=train_params["data.num_workers"],
        collate_fn=CollateDefault(),
    )

    model = LitClassifier(model_dir=MODEL_DIR, backbone=Backbone(hidden_dim=128), learning_rate=train_params["opt.lr"])
    trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        strategy=train_params["trainer.strategy"],
        devices=train_params["trainer.num_devices"],
        # gpus=train_params["trainer.num_devices"],
        num_nodes=train_params["trainer.num_nodes"]
        # auto_select_gpus=True,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    check_nvidia_smi()

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=32)
    args = parser.parse_args()

    # override train_params with args captured
    TRAIN_PARAMS["trainer.num_nodes"] = args.num_nodes
    TRAIN_PARAMS["trainer.num_devices"] = args.num_devices
    TRAIN_PARAMS["opt.lr"] = args.lr
    TRAIN_PARAMS["data.batch_size"] = args.bs

    # check if we should enable ddp
    use_ddp = False
    if args.num_nodes > 1 or args.num_devices > 1:
        use_ddp = True
    TRAIN_PARAMS["trainer.strategy"] = "ddp" if use_ddp else None

    # start training
    run_train(paths=PATHS, train_params=TRAIN_PARAMS)
