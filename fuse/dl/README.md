# FuseMedML DL package

The fuse.dl package contains core PyTorch-based DL related modules that make heavy use of existing DL libraries. Currently, PyTorch and PyTorch Lightning are the two primary frameworks that FuseMedML is based on.
The fuse.dl facilitates using PyTorch-based models and training process within FuseMedML. It also provides generic components based on FuseMedML concepts, that can be used in your DL pipeline. They can also be combined with components from other FuseMedML packaged such as fuse.data or fuse.eval.

FuseMedML is very flexible. This means that a user can choose to only use some components from fuse.dl or none at all, and only use other FuseMedML packages. With respect to the training loop, fuse.dl offers two levels of customization based on PyTorch Lightning as explained below. However, a user may opt for pure PyTorch based training loop, or use another high-level library for it like PyTorch Ignite.

## lightning
FuseMedML uses PyTorch Lightning as it's model training "manager".
There are two main supported ways in which lightning is used in fuse:
### 1. Ready-made wrapper use (suitable mostly for supervised learning use-cases)
(see [MNIST example](../../fuse_examples/imaging/classification/mnist/simple_mnist_starter.py)).
In this mode, we use `LightningModuleDefault`, a FuseMedML class that inherits from PyTorch Lightning's `LightningModule` class. It is instantiated as follows:

```
pl_module = LightningModuleDefault(model_dir=model_dir,
                                       model=model,
                                       losses=losses,
                                       train_metrics=train_metrics,
                                       validation_metrics=validation_metrics,
                                       best_epoch_source=best_epoch_source,
                                       optimizers_and_lr_schs=optimizers_and_lr_schs)
```
where the user is only responsible to supply the required arguments, including the model, losses, train and validation metrics, all of which are in fuse style (classes implemented or supported in FuseMedML).


### 2. Custom use (suitable for any use case)
(see [MNIST example](../../fuse_examples/imaging/classification/mnist/run_mnist_custom_pl_imp.py))

In this more flexible and custom mode, the user is responsible to implement their own `LightningModule` tailored to their project, with all standard methods required by PyTorch Lightning such as `forward`, `training_step` etc'.
Once such a custom module exists, it is instantiated with a user defined set of arguments:
```
pl_module = CustomLightningModule(**custom_args)
```

`pl_funcs` contains a collection of helpful functions that can be used in implementing `CustomLightningModule`.

## losses
This module implements basic FuseMedML loss classes as well as a few specific loss function implementations.
The basic classes are:
1. `LossBase` - A generic template class that can be used for implementing any loss function. All it does is receives a `batch_dict` and returns a loss tensor.
2. `LossWarmUp` - A wrapper class that gets an existing loss function and zeros its value for a given number of iterations. This is useful in a multiple loss function scenario, where one loss should be stabilized before the other one comes into play.
3. `LossDefault` - Default loss implementation of FuseMedML. It is still a generic class in the sense that is can operate on any actual loss function implementation. In addition, it can receive a scalar multiplier value, and a custom preprocessing function that operated on the `batch_dict` before the loss operates on it.
[The MNIST example uses `LossDefault`](../../fuse_examples/imaging/classification/mnist/simple_mnist_starter.py) as follows:

```
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
        ),
    }
```

The specific losses currently implemented are suitable for classification and dense segmentation tasks:
1. `LossSegmentationCrossEntropy` - Cross-entropy loss with optional class weights and resizing of the ground truth map.
2. `BinaryDiceLoss` - Dice loss with optional class weights, `batch_dict` preprocessing and ground truth map resizing.
3. `FocalLoss` - Focal loss with configurable hyperparameters and optional scalar weight, `batch_dict` preprocessing and ground truth map resizing.

Note again, that while these are specific loss functions that are implemented, it is possible to use any loss, whether custom, or one that already exists in PyTorch, and pass it to the `LossDefault` class for use in FuseMedML.

## models
This module contains DL model and architecture related classes. A FuseMedML model is very similar to a PyTorch model. The only difference is that in its `forward` function it gets as input a fuse `batch_dict`, extracts the input data from the input keys, forwards it through the model and writes the output to the `batch_dict`'s output keys.
The most basic class is `ModelWrapSeqToDict`. It is FuseMedML's wrapper for PyTorch models to be used in fuse. It is initialized with an existing PyTorch model, and a list of input and output keys.
Optionally, a pre and post processing function can be provided to be applied on the `batch_dict` before and after model run.
[The MNIST example uses `ModelWrapSeqToDict`](../../fuse_examples/imaging/classification/mnist/simple_mnist_starter.py) to wrap a PyTorch model as follows:

```
model = ModelWrapSeqToDict(
        model=torch_model,
        model_inputs=["data.image"],
        post_forward_processing_function=perform_softmax,
        model_outputs=["model.logits.classification", "model.output.classification"],
    )
```

 Additional basic classes include:
 1. `ModelEnsemble` - Initialized from a list of trained models directory, this class can run several models sequentially. It then produces a dictionary with predictions of each model in the ensemble, as well as average and majority vote over the predictions.
 2. `ModelMultiHead` - A class which given a backbone (feature extractor network) and multiple heads (shallow networks that output desired final results such as global/coarse or dense prediction), implements a neural network with the corresponding structure.
 3. `ModelMultistream` - Implements a neural network with multiple backbone processing streams and heads.
 4. `ModelSiamese` - Implements a siamese neural network, with two identical backbones and multiple heads.

Besides these basic classes, more specific architecture blocks are implemented. They are divided into "backbones" and "heads".

Implemented backbones include a "vanilla" fully connected network, or Multi Layer Perceptron (MLP), supported versions of 2D and 3D ResNets, and an Inception ResNet.  

Implemented "heads" include a number of parameterized classifier heads, in 1D, 2D and 3D, as well as a dense segmentation head.

[The STOIC 21 example](../../fuse_examples/imaging/classification/stoic21/runner_stoic21.py) uses `ModelMultiHead`, a 3D ResNet backbone `BackboneResnet3d`, and a 3D classification head `Head3D` as follows:
 ```
 model = ModelMultiHead(
        conv_inputs=(("data.input.img", 1),),
        backbone=BackboneResnet3D(in_channels=1),
        heads=[
            Head3D(
                head_name="classification",
                mode="classification",
                conv_inputs=[("model.backbone_features", 512)],
                dropout_rate=imaging_dropout,
                append_dropout_rate=clinical_dropout,
                fused_dropout_rate=fused_dropout,
                num_outputs=2,
                append_features=[("data.input.clinical", 8)],
                append_layers_description=(256, 128),
            ),
        ],
    )
 ```
These building blocks are also similarly reused in the [KNIGHT challenge example](../../fuse_examples/imaging/classification/knight/baseline/fuse_baseline.py).


## templates
This module contains a walkthrough template code rich with comments, to demonstrate training with FuseMedML, with all required building blocks.
