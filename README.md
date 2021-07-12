[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg)](https://opensource.org/)
[![PyPI version](https://badge.fury.io/py/fuse-med-ml.svg)](https://badge.fury.io/py/fuse-med-ml)
[![Slack channel](https://img.shields.io/badge/support-slack-slack.svg?logo=slack)](https://join.slack.com/t/newworkspace-i3g4445/shared_invite/zt-sr0hcb9f-E~SLYbG9bE5fn8iq5OE0ww)

<img src="fuse/doc/FuseMedML-logo.png" alt="drawing" width="30%"/>

# What is FuseMedML?
FuseMedML is an **open-source PyTorch based framework designed to facilitate deep learning R&D in medical imaging**.
FuseMedML provides entire built-in pipeline, from data processing through train and inference to analyze and visualize.

# Why use FuseMedML?
FuseMedML punctilious software design provides many advantages, making it a great framework for deep-learning research and development in medical imaging:
* **Rapid development** -

  Requires implementing only minimal scope of code to get up and running pipeline.
  Common generic implementation of most components in pipeline is provided and only specific components such as data extractor expected to be implemented by the user.
* **Flexible, customizable, and scalable** -

  The default implementation of modules and components suits for many common cases, 
  However, components in the pipeline kept decoupled, allowing a user to re-implement a component to get the required behavior.
* **Encourage sharing** - 

  Sharing is a powerful tool. To encourage sharing, FuseMedML design enables to implement generic components which can be easily used in Plug & Play manner.
* **Common easy to use generic components and examples** - 

  Such as: Caching, Augmentor, Visualizer, Sampler/Balancer, Metrics, Losses, etc.
* **Medical Imaging Expertise** - 

  The pre-implemented components implemented by a group of medical-imaging researchers and tackles many of the challenges in this field.   

# Installation
The best way of installing `FuseMedML` is by using `pip`:
```bash
$ pip install fuse-med-ml
```
 FuseMedML supports Python 3.6 or later and PyTorch 1.5 or later. Full dependencies list can be found in [**requirements.txt**](requirements.txt).
 
An alternative, allowing to edit the source code, would be to download the repo and install it using:
```bash
$ pip install -e .
```

# Ready to get going?
## FuseMedML from the ground up
[**User Guide**](fuse/doc/user_guide.md) - including detailed explanation about FuseMedML modules, structure, concept. etc.

[**High Level Code Example**](fuse/doc/high_level_example.md) - example for binary classifier for mammography with an auxiliary segmentation loss and clinical data

## Examples
* classification
    * [**MNIST**](fuse/examples/classification/mnist/)  - a simple example, including training, inference and evaluation over [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    * [**Skin Lesion**](fuse/examples/classification/skin_lesion/) - skin lesion classification , including training, inference and evaluation over the public dataset introduced in [ISIC challenge](https://challenge.isic-archive.com/landing/2017)
    * [**Prostate Gleason Classifiaction**](fuse/examples/classification/prostate_x/) - lesions classification of Gleason score in prostate over the public dataset introduced in [SPIE-AAPM-NCI PROSTATEx challenge](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38)

## Walkthrough template
* [**Walkthrough Template**](fuse/templates/walkthrough_template.py) - the template includes several TODO notes, marking the minimal scope of code required to get up and running pipeline. The template also includes useful explanation and tips.

## Community support
We use the Slack workspace at fusemedml.slack.com for informal communication.
We encourage you to ask questions regarding FuseMedML that don't necessarily merit opening an issue on Github.

[**Use this invite link to join FuseMedML on Slack**](https://join.slack.com/t/newworkspace-i3g4445/shared_invite/zt-sr0hcb9f-E~SLYbG9bE5fn8iq5OE0ww).

IBMers can also join a slack channel on IBM Research organization:
[**#fusers**](https://ibm-research.slack.com/archives/C0176S37QNP) .


