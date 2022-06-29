# FuseMedML DL package

The fuse.dl package contains core DL related modules that make heavy use of existing DL libraries. Currently, PyTorch and PyTorch Lightning are the two primary frameworks that FuseMedML is based on.

## lightning
FuseMedML uses Pytorch Lightning as it's model training "manager".  
There are two main supported ways in which lightning is used in fuse:
1. Ready-made wrapper use (suitable mostly for supervised learning use-cases)
(see [MNIST example](../examples/fuse_examples/imaging/classification/mnist/run_mnist.py))
    
2. Custom use (suitable for any use case)
(see [MNIST example](../examples/fuse_examples/imaging/classification/mnist/run_mnist_custom_pl_imp.py))


Note: previously, a dedicated PyTorch based [manager](manager) was used which is now deprecated. Some of the implemented [examples](../examples) still use it but we plan to port them to use the lightning manager.

## losses

## models

## optimizers

## templates
This module contains a walkthrough template code rich with comments, to demonstrate training with FuseMedML, with all required building blocks.
