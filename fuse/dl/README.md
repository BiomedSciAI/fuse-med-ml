# FuseMedML DL package

The fuse.dl package contains core PyTorch-based DL related modules that make heavy use of existing DL libraries. Currently, PyTorch and PyTorch Lightning are the two primary frameworks that FuseMedML is based on.  
The fuse.dl facilitates using PyTorch-based models and training process within FuseMedML. It also provides generic components based on FuseMedML concepts, that can be used in your DL pipeline. They can also be combined with components from other FuseMedML packaged such as fuse.data or fuse.eval.

## lightning
FuseMedML uses PyTorch Lightning as it's model training "manager".  
There are two main supported ways in which lightning is used in fuse:
### 1. Ready-made wrapper use (suitable mostly for supervised learning use-cases)
(see [MNIST example](../../examples/fuse_examples/imaging/classification/mnist/run_mnist.py)).  
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
(see [MNIST example](../../examples/fuse_examples/imaging/classification/mnist/run_mnist_custom_pl_imp.py))  

In this more flexible and custom mode, the user is responsible to implement their own `LightningModule` tailored to their project, with all standard methods required by PyTorch Lightning such as `forward`, `training_step` etc'.  
Once such a custom module exists, it is instantiated with a user defined set of arguments:
```
pl_module = CustomLightningModule(**custom_args)
```  

Note: previously, a dedicated PyTorch based [manager](managers) was used which is now deprecated. Some of the implemented [examples](../../examples) still use it but we plan to port them to use the lightning manager.

## losses
This module implements basic FuseMedML loss classes as well as a few specific loss function implementations.
The basic classes are:  
1. `LossBase` - A generic template class that can be used for implementing any loss function. All it does is receives a `batch_dict` and returns a loss tensor.
2. `LossWarmUp` - A wrapper class that gets an existing loss function and zeros its value for a given number of iterations. This is useful in a multiple loss function scenario, where one loss should be stabilized before the other one comes into play.
3. `LossDefault` - Default loss implementation of FuseMedML. It is still a generic class in the sense that is can operate on any actual loss function implementation. In addition, it can receive a scalar multiplier value, and a custom preprocessing function that operated on the `batch_dict` before the loss operates on it.  
[The MNIST example uses `LossDefault`](../../examples/fuse_examples/imaging/classification/mnist/run_mnist.py)

The specific losses currently implemented are suitable for classification and dense segmentation tasks:  
1. `LossSegmentationCrossEntropy` - Cross-entropy loss with optional class weights and resizing of the ground truth map.
2. `BinaryDiceLoss` - Dice loss with optional class weights, `batch_dict` preprocessing and ground truth map resizing. 
3. `FocalLoss` - Focal loss with configurable hyperparameters and optional scalar weight, `batch_dict` preprocessing and ground truth map resizing.  

Note again, that while these are specific loss functions that are implemented, it is possible to use any loss, whether custom, or one that already exists in PyTorch, and pass it to the `LossDefault` class for use in FuseMedML.

## models
This module contains DL model and architecture related classes. 
The most basic class is `ModelWrapSeqToDict`. It is FuseMedML's wrapper for PyTorch models to be used in fuse. It is initialized with an existing PyTorch module, and a list of input and output keys. When `forward` is called on a fuse `batch_dict`, it extracts the data from the input keys from `batch_dict`, calls the model `forward` function on the data, and writes the output to the `batch_dict`'s output keys.  
Optionally, a pre and post processing function can be provided to be applied on the `batch_dict` before and after model run.  
[The MNIST example uses `ModelWrapSeqToDict`](../../examples/fuse_examples/imaging/classification/mnist/run_mnist.py)

 Additional basic classes include:
 1. `ModelEnsemble` - Initialized from a list of trained models directory, this class can run several models sequentially. It then produces a dictionary with predictions of each model in the ensemble, as well as average and majority vote over the predictions.
 2. `ModelMultiHead` - A class which given backbone and multiple heads, implements a neural network with the corresponding structure.  
 [The STOIC 21 example uses `ModelMultiHead`](../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py)  
 3. `ModelMultistream` - Implements a neural network with multiple backbone processing streams and heads.
 4. `ModelSiamese` - Implements a siamese neural network, with two identical backbones and multiple heads.

Besides these basic classes, more specific architecture blocks are implemented. They are divided into "backbones" and "heads".

Implemented backbones include a "vanilla" fully connected network, or Multi Layer Perceptron (MLP), supported versions of 2D and 3D ResNets, and an Inception ResNet.  
[The KNIGHT challenge example uses a 3D ResNet backbone `BackboneResnet3D`](../../examples/fuse_examples/imaging/classification/knight/baseline/fuse_baseline.py)  

Implemented "heads" include a number of parameterized classifier heads, in 1D, 2D and 3D, as well as a dense segmentation head.  
[The KNIGHT challenge example uses a 3D classification head `Head3DClassifier`](../../examples/fuse_examples/imaging/classification/knight/baseline/fuse_baseline.py)

## optimizers
This module includes an implementation of the [SAM](https://github.com/davda54/sam) optimizer, and a callback function for optimizers that require a closure argument.

## templates
This module contains a walkthrough template code rich with comments, to demonstrate training with FuseMedML, with all required building blocks.
