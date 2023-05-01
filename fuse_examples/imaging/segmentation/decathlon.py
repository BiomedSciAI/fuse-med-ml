from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    Orientationd,
    Randomizable,
    Resized,
    ScaleIntensityd,
    Spacingd,
    EnsureTyped,
)
# from monai.networks.nets import UNet, DenseNet121
from monai.losses import DiceLoss
from monai.data import DataLoader
from monai.config import print_config
from monai.apps import DecathlonDataset

import torch.optim as optim
import matplotlib.pyplot as plt
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.losses import LossDefault
from pytorch_lightning import Trainer
from unet import UNet

transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys="image"),
        Resized(keys=["image", "label"], spatial_size=(32, 64, 32), mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ]
)

root_dir = "/dccstor/mm_hcls/datasets/msd"
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task04_Hippocampus",
    transform=transform,
    section="training",
    download=False,
)
validation_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task04_Hippocampus",
    transform=transform,
    section="validation",
    download=False,
)
# the dataset can work seamlessly with the pytorch native dataset loader,
# but using monai.data.DataLoader has additional benefits of mutli-process
# random seeds handling, and the customized collate functions
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=16)
validation_loader = DataLoader(validation_ds, batch_size=32, shuffle=True, num_workers=16)
model = UNet(
    input_name="image",
    seg_name="label",
    pre_softmax="model.logits.all_pred_target",
    post_softmax="model.outputs.all_pred_target",
    out_features=2,
    unet_kwargs={
    "strides" : [[2, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
    "channels" : [32, 64, 128, 256, 512, 1024],
    "out_channels" : 2 ,
    "in_channels" : 1,
    "num_res_units": 2,
    "spatial_dims" : 3}
)
model_dir="/dccstor/mm_hcls/guez/fuse-med-ml/working_dir/model_dir"
losses = {}
losses["segmentation"] = LossDefault(
        pred="image",
        target="label",
        callable=DiceLoss(to_onehot_y=True, softmax=True),
        weight=1.0)
best_epoch_source = dict(
        monitor="validation.losses.total_loss",  #metrics.auc.macro_avg",
        mode="min",
    )
train = {"learning_rate" : 1e-4, "weight_decay" : 1e-4 , "trainer" :{"num_epochs": 10,"accelerator": 'gpu' , 'devices': 1, "ckpt_path":model_dir+"/best.ckpt"} }
optimizer = optim.Adam(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

lr_sch_config = dict(scheduler=scheduler, monitor="validation.losses.total_loss")

# optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)
pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics={},
        validation_metrics={},
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

model_config={

}
pl_trainer = Trainer(
    replace_sampler_ddp=False,  # Must be set when using a batch sampler
        default_root_dir=model_dir,
        max_epochs=train["trainer"]["num_epochs"],
        accelerator=train["trainer"]["accelerator"],
        devices=train["trainer"]["devices"],
        num_sanity_val_steps=-1,
        auto_select_gpus=True,
)

pl_trainer.fit(pl_module, train_loader, validation_loader, ckpt_path=None)